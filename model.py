from torch import nn, Tensor
import torch
from tqdm import tqdm
import math
import numpy as np
import random
import copy


# Model
class rnn_model(torch.nn.Module):
    def __init__(self, rank, dims, hidden_size):
        super(rnn_model, self).__init__()
        self.rank = rank
        self.order = len(dims)
        self.layer_first = nn.Linear(hidden_size, rank)
        self.layer_middle = nn.Linear(hidden_size, rank*rank)
        self.layer_final = nn.Linear(hidden_size, rank)                
        self.rnn = nn.LSTM(hidden_size, hidden_size)                    
        self.hidden_size = hidden_size        
                
        num_emb = np.sum(dims)
        self.emb = nn.Embedding(num_embeddings=num_emb, embedding_dim = hidden_size)        
        
        
    '''
        _input: batch size x seq len        
       -----------------------------------
       preds: batch size 
    '''
    def forward(self, _input):
        _input = _input.transpose(0, 1)
        seq_len, batch_size = _input.size()
        _input = self.emb(_input)   # seq len x batch size x hidden dim                    
        rnn_output, _ = self.rnn(_input)   # seq len x batch size x hidden dim 

        first_mat = self.layer_first(rnn_output[0,:,:])   
        final_mat = self.layer_final(rnn_output[-1,:,:])   # batch size x R
        first_mat, final_mat = first_mat.unsqueeze(1), final_mat.unsqueeze(-1)   # batch size x 1 x R, batch size x R x 1
        middle_mat = self.layer_middle(rnn_output[1:-1,:,:])  # seq len -2 x batch size x R^2        
        middle_mat = middle_mat.view(self.order-2, batch_size, self.rank, self.rank)  # seq len - 2  x batch size x R x R
        
        preds = torch.matmul(first_mat, middle_mat[0, :, :, :])
        for j in range(1, self.order-2):
            preds = torch.matmul(preds, middle_mat[j, :, :, :])
        preds = torch.matmul(preds, final_mat).squeeze()  # batch size 
        return preds
        
        
        
class TensorCodec:
    
    '''
        input_size: list of list that saves the size of inputs of all levels for each mode,
        order x k 
    '''
    def __init__(self, input_mat, rank, hidden_size, device):
        # Intialize parameters
        self.input_mat = input_mat
        self.dims = input_mat.dims
        self.order = input_mat.order
        self.hidden_size = hidden_size
        self.device = device
        self.i_device = torch.device("cuda:" + str(self.device[0]))
        self.model = rnn_model(rank, self.dims, hidden_size)
        self.model.double()     
        if len(self.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids = self.device)                        
        self.model = self.model.to(self.i_device)  
                            
        # Build add term 
        self._add = []
        temp_dim = 0
        for i in range(self.order):            
            self._add.append(temp_dim)
            temp_dim += self.dims[i]
        
        #print(self._add)    
        # move to gpu        
        self._add = torch.tensor(self._add, dtype=torch.long, device=self.i_device).unsqueeze(0)                
        self.comp_size =  sum(p.numel() for p in self.model.parameters() if p.requires_grad) * 8       
        print(f"Compressed size:{self.comp_size} bytes")
        
    
    # Given a model indices output predictions
    # model_idx: batch size x order
    # output: batch size
    def predict(self, model_input):        
        model_input = model_input + self._add        
        return self.model(model_input)

    
    # Define L2 loss
    def L2_loss(self, is_train, batch_size):                
        return_loss = 0.
        for i in tqdm(range(0, self.input_mat.num_entries, batch_size)):
            with torch.no_grad():
                curr_batch_size = min(batch_size, self.input_mat.num_entries - i)
                curr_ten_idx = torch.arange(i, i + curr_batch_size, dtype=torch.long, device = self.i_device)
                curr_ten_idx = curr_ten_idx.unsqueeze(-1) // self.input_mat.src_base % self.input_mat.dims_gpu # batch size x order      
                            
            preds = self.predict(curr_ten_idx)               
            vals = torch.tensor(self.input_mat.vals[i:i+curr_batch_size], device=self.i_device)                                   
            curr_loss = torch.square(preds - vals).sum()                   
            return_loss += curr_loss.item()
                        
            if is_train:
                curr_loss.backward()                
        #print(f'root val sum: {math.sqrt(val_sum)}, loss:{math.sqrt(return_loss)}, pred sum:{pred_sum}')
        return return_loss    
            
        
    # minibatch L2 loss
    # samples: indices of sampled matrix entries
    def L2_minibatch_loss(self, batch_size, samples):
        return_loss, minibatch_norm = 0., 0.
        num_sample = samples.shape[0]
        # Indices of sampled matrix entries        
        for i in range(0, num_sample, batch_size):
            with torch.no_grad():
                curr_batch_size = min(batch_size, num_sample - i)
                curr_ten_idx = samples[i:i+curr_batch_size]
                vals = torch.tensor(self.input_mat.vals[curr_ten_idx], device=self.i_device)
                
                curr_ten_idx = torch.tensor(curr_ten_idx, device=self.i_device).unsqueeze(-1)
                curr_ten_idx = curr_ten_idx // self.input_mat.src_base % self.input_mat.dims_gpu # batch size x self.order
                                                
            preds = self.predict(curr_ten_idx)
            curr_loss = torch.square(preds - vals).sum()
            return_loss += curr_loss.item()
            minibatch_norm += torch.square(vals).sum().item()
            curr_loss.backward()