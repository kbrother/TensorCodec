from torch import nn
import torch
from tqdm import tqdm
import math
import numpy as np

# Model
class rnn_model(torch.nn.Module):
    '''
        m_list: save the number of rows in the input
        n_list: save the number of columns in the input
    '''
    def __init__(self, rank, m_list, n_list, hidden_size):
        super(rnn_model, self).__init__()
        self.rank = rankz
        self.k = len(n_list)
        self.linear_first = nn.Linear(hidden_size, rank)
        self.linear_final = nn.Linear(hidden_size, rank)
        self.linear_middle = nn.Linear(hidden_size, rank*rank)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        
        mn_set = set()
        for i in range(self.k):
            mn_set.add((m_list[i], n_list[i]))        
        num_emb = 0
        for mn in mn_set:
            num_emb += mn[0]*mn[1]
        self.emb = nn.Embedding(num_embeddings=num_emb, embedding_dim = hidden_size)
        
    '''
        _input: batch size x seq len        
    '''
    # return value: batch size
    def forward(self, _input):
        _input = _input.transpose(0, 1)
        _, batch_size = _input.size()
        _input = self.emb(_input)   # batch size x seq len x hidden dim        
        
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(_input)   # seq len x batch size x hidden dim        
        first_mat = self.linear_first(rnn_output[0,:,:])   
        final_mat = self.linear_final(rnn_output[-1,:,:])
        first_mat, final_mat = first_mat.unsqueeze(1), final_mat.unsqueeze(-1)   # batch size x 1 x R
        middle_mat = self.linear_middle(rnn_output[1:-1,:,:]) 
        #print(middle_mat.shape)
        middle_mat = middle_mat.view(self.k-2, batch_size, self.rank, self.rank)  # seq len - 2  x batch size x R x R
        
        _output = torch.matmul(first_mat, middle_mat[0, :, :, :])
        for i in range(1, self.k-2):
            _output = torch.matmul(_output, middle_mat[i, :, :, :])
        _output = torch.matmul(_output, final_mat)  # batch size         
        return _output
    
    
# Tensor train decomposition
class TT: 
    def __init__(self, input_mat, rank, m_list, n_list, device, dataset):
        self.k = len(m_list)
        self.device = torch.device("cuda:" + str(device))
        data_folder = "TTD/" + dataset + "/"
        self.cores = []
        for i in range(self.k):
            curr_core = np.load(data_folder + str(i+1) + ".npy")
            curr_core = curr_core.astype(np.double)
            self.cores.append(torch.tensor(curr_core, device=self.device))    
        self.input_mat = input_mat

        
        # Build bases
        self.row_bases, self.col_bases = [], []
        _base = 1
        for i in range(self.k-1, -1, -1):
            self.row_bases.insert(0, _base)
            _base *= m_list[i]
        _base = 1
        for i in range(self.k-1, -1, -1):
            self.col_bases.insert(0, _base)
            _base *= n_list[i]
            
        self.m_list = torch.tensor(m_list, dtype=torch.long, device=device).unsqueeze(0)
        self.n_list = torch.tensor(n_list, dtype=torch.long, device=device).unsqueeze(0)
        self.row_bases = torch.tensor(self.row_bases, dtype=torch.long, device=self.device)
        self.col_bases = torch.tensor(self.col_bases, dtype=torch.long, device=self.device)
        
    def fitness(self, batch_size):
        with torch.no_grad():
            sq_err = 0.
            for i in tqdm(range(0, self.input_mat.num_entries, batch_size)):
                with torch.no_grad():
                    curr_batch_size = min(batch_size, self.input_mat.num_entries - i)
                    curr_idx = torch.arange(i, i + curr_batch_size, dtype=torch.long, device = self.device)
                    row_idx, col_idx = curr_idx // self.input_mat.num_col, curr_idx % self.input_mat.num_col
                    row_idx = row_idx.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
                    col_idx = col_idx.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k          
                    
                    preds = self.cores[0][row_idx[:,0], col_idx[:,0], :, :]   # batch size x 1 x R                         
                    #print(preds[1,:,:])
                    for j in range(1, self.k):
                        curr_core = self.cores[j][row_idx[:,j], col_idx[:,j], :, :]    # batch size x R x R                                    
                        preds = torch.matmul(preds, curr_core)  
                        #print(curr_core[1,:,:])
                    preds = preds.squeeze()   
                    vals = torch.tensor(self.input_mat.vals[i:i+curr_batch_size], device=self.device, dtype=torch.double)                    
                    sq_err += torch.square(preds - vals).sum().item()
                    #print(preds)
                    #print(vals)
                    #break
                    
        return 1 - math.sqrt(sq_err)/self.input_mat.norm
    
class NeuKron_TT:
    '''
        m_list: save the number of rows in the input, (should be increasing order)
        n_list: save the number of columns in the input, (should be increasing order)
    '''
    def __init__(self, input_mat, rank, m_list, n_list, hidden_size, device):
        # Intialize parameters
        self.input_mat = input_mat
        self.m_list, self.n_list = m_list, n_list
        self.k = len(m_list)
        self.hidden_size = hidden_size
        self.device = device
        self.i_device = torch.device("cuda:" + str(self.device[0]))
        self.model = rnn_model(rank, m_list, n_list, hidden_size)
        self.model.float()
        if len(self.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids = self.device)
        self.model = self.model.to(self.i_device)
        
        # Build bases
        self.row_bases, self.col_bases = [], []
        _base = 1
        for i in range(self.k-1, -1, -1):
            self.row_bases.insert(0, _base)
            _base *= m_list[i]
        _base = 1
        for i in range(self.k-1, -1, -1):
            self.col_bases.insert(0, _base)
            _base *= n_list[i]
    
        # Build add term 
        mn_dict = {}        
        _temp = 0
        for i in range(self.k):
            if (m_list[i], n_list[i]) not in mn_dict:
                mn_dict[(m_list[i], n_list[i])] = _temp
                _temp += m_list[i] * n_list[i]
        self._add = []
        for i in range(self.k):
            self._add.append(mn_dict[(m_list[i], n_list[i])])
                                    
        # move to gpu
        self.m_list = torch.tensor(m_list, dtype=torch.long, device=self.i_device)
        self.n_list = torch.tensor(n_list, dtype=torch.long, device=self.i_device)
        self.row_bases = torch.tensor(self.row_bases, dtype=torch.long, device=self.i_device)
        self.col_bases = torch.tensor(self.col_bases, dtype=torch.long, device=self.i_device)
        self._add = torch.tensor(self._add, dtype=torch.long, device=self.i_device).unsqueeze(0)                
        
        print(f"The number of params:{ sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
    # Define L2 loss
    def L2_loss(self, is_train, batch_size):                
        loss = 0.
        for i in range(0, self.input_mat.real_num_entries, batch_size):
            with torch.no_grad():
                curr_batch_size = min(batch_size, self.input_mat.real_num_entries - i)
                curr_idx = torch.arange(i, i + curr_batch_size, dtype=torch.long, device = self.i_device)
                row_idx, col_idx = curr_idx // self.input_mat.src_ncol, curr_idx % self.input_mat.src_ncol
                row_idx = row_idx.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
                col_idx = col_idx.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k                
                _input = row_idx * self.n_list + col_idx + self._add
                
            preds = self.model(_input)
            vals = torch.tensor(self.input_mat.src_vals[i:i+curr_batch_size], device=self.i_device)
            curr_loss = torch.square(preds - vals).sum()
            loss += curr_loss.item()
            if is_train:
                curr_loss.backward()
        return loss                