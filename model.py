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
        self.rank = rank
        self.k = len(n_list)
        self.layer_first = nn.Linear(hidden_size, rank)
        self.layer_middle = nn.Linear(hidden_size, rank*rank)
        self.layer_final = nn.Linear(hidden_size, rank)        
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
        first_mat: batch size x 1 x R
        middle_mat: seq_len -2 x batch size x R x R
        final_mat: batch size x R x 1
    '''
    def forward(self, _input):
        _input = _input.transpose(0, 1)
        _, batch_size = _input.size()
        _input = self.emb(_input)   # batch size x seq len x hidden dim        
        
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(_input)   # seq len x batch size x hidden dim        
        first_mat = self.layer_first(rnn_output[0,:,:])   
        final_mat = self.layer_final(rnn_output[-1,:,:])   # batch size x R
        first_mat, final_mat = first_mat.unsqueeze(1), final_mat.unsqueeze(-1)   # batch size x 1 x R, batch size x R x 1
        middle_mat = self.layer_middle(rnn_output[1:-1,:,:])  # seq len -2 x batch size x R^2
        #print(middle_mat.shape)
        middle_mat = middle_mat.view(self.k-2, batch_size, self.rank, self.rank)  # seq len - 2  x batch size x R x R
                                                          
        preds = torch.matmul(first_mat, middle_mat[0, :, :, :])
        for j in range(1, self.k-2):
            preds = torch.matmul(preds, middle_mat[j, :, :, :])
        preds = torch.matmul(preds, final_mat).squeeze()  # batch size 
        return preds
    
# Tensor train decomposition
class TT: 
    def __init__(self, input_mat, rank, m_list, n_list, device, dataset):
        self.k = len(m_list)
        self.device = torch.device("cuda:" + str(device))
        data_folder = "TTD/" + dataset + "/rank" + str(rank) + "/"
        self.cores = []
        for i in range(self.k):
            curr_core = np.load(data_folder + str(i+1) + ".npy")  # m x n x R x R
            curr_core = curr_core.astype(np.double)
            self.cores.append(torch.tensor(curr_core, device=self.device))    # 
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
                    for j in range(1, self.k):
                        curr_core = self.cores[j][row_idx[:,j], col_idx[:,j], :, :]    # batch size x R x R                                    
                        preds = torch.matmul(preds, curr_core)  
                        
                    preds = preds.squeeze()   
                    vals = torch.tensor(self.input_mat.vals[i:i+curr_batch_size], device=self.device, dtype=torch.double)                    
                    sq_err += torch.square(preds - vals).sum().item()        
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
        self.model.double()        
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
        return_loss = 0.
        val_sum = 0.
        pred_sum = 0.
        for i in range(0, self.input_mat.real_num_entries, batch_size):
            with torch.no_grad():
                curr_batch_size = min(batch_size, self.input_mat.real_num_entries - i)
                curr_idx = torch.arange(i, i + curr_batch_size, dtype=torch.long, device = self.i_device)
                row_idx, col_idx = curr_idx // self.input_mat.src_ncol, curr_idx % self.input_mat.src_ncol
                row_idx = row_idx.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
                col_idx = col_idx.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k                
                _input = row_idx * self.n_list + col_idx + self._add
                
            #_input = _input.cpu()
            #self.model.to(torch.device("cpu"))            
            preds = self.model(_input).squeeze()  # batch size                    
            vals = torch.tensor(self.input_mat.src_vals[i:i+curr_batch_size], device=self.i_device)                                   
            curr_loss = torch.square(preds - vals).sum()                   
            return_loss += curr_loss.item()
            #pred_sum += torch.square(preds).sum().item()
            #val_sum += torch.square(vals).sum().item()
                        
            if is_train:
                curr_loss.backward()                
        #print(f'root val sum: {math.sqrt(val_sum)}, loss:{math.sqrt(return_loss)}, pred sum:{pred_sum}')
        return return_loss    
    
    
    # Explicitly guided loss using TTD of which the output has the same size
    def L2_guide_loss(self, is_train, ttd, batch_size):
        loss = 0.
        for i in range(0, self.input_mat.real_num_entries, batch_size):
            with torch.no_grad():
                curr_batch_size = min(batch_size, self.input_mat.real_num_entries - i)
                curr_idx = torch.arange(i, i + curr_batch_size, dtype=torch.long, device = self.i_device)
                row_idx, col_idx = curr_idx // self.input_mat.src_ncol, curr_idx % self.input_mat.src_ncol
                row_idx = row_idx.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
                col_idx = col_idx.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k                
                _input = row_idx * self.n_list + col_idx + self._add
            
            first_pred, middle_pred, final_pred = self.model(_input)
            first_vals = ttd.cores[0][row_idx[:, 0], col_idx[:, 0], :, :] # batch size x 1 x R
            middle_vals = [ttd.cores[j][row_idx[:, j], col_idx[:, j], :, :].unsqueeze(0) for j in range(1, self.k-1)]
            middle_vals = torch.cat(middle_vals, 0) # seq len-2 x batch size R x R
            final_vals = ttd.cores[self.k-1][row_idx[:, self.k-1], col_idx[:, self.k-1], :, :] # batch size x R x 1
            
            curr_loss = torch.square(first_pred - first_vals).sum()
            curr_loss = curr_loss + torch.square(middle_pred - middle_vals).sum()
            curr_loss = curr_loss + torch.square(final_pred - final_vals).sum()
            if is_train:
                curr_loss.backward()
            loss += curr_loss.item()
        return loss
    
    '''
        Compare the input and output        
    '''
    def check_output(self, row_idx, col_idx, ttd):        
        with torch.no_grad():
            self.model.eval()
            row_idx = torch.tensor(row_idx, dtype=torch.long, device=self.i_device)
            col_idx = torch.tensor(col_idx, dtype=torch.long, device=self.i_device)
            row_idx = row_idx.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
            col_idx = col_idx.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k                
            _input = row_idx * self.n_list + col_idx + self._add
            
            first_pred, middle_pred, final_pred = self.model(_input)
            first_vals = ttd.cores[0][row_idx[0], col_idx[0], :, :] # batch size x 1 x R
            print(f'first pred: {first_pred}, first sol: {first_vals}')
            
            middle_vals = [ttd.cores[j][row_idx[j], col_idx[j], :, :].unsqueeze(0) for j in range(1, self.k-1)]            
            for j in range(1, self.k-1):
                print(f'middel pred:{middle_pred[j-1,:,:,:]}, middle sol: {ttd.cores[j][row_idx[j], col_idx[j], :, :]}')
            final_vals = ttd.cores[self.k-1][row_idx[self.k-1], col_idx[self.k-1], :, :] # batch size x R x 1
            print(f'final pred: {final_pred}, final sol:{final_vals}')
            
    
        '''
        Input
            curr_order: 0 for row and 1 for col
            model_idx: indices of model
            num_bucket: number of bucket
            
        Output
            bucket_idx: buckets correspond to model indices
    '''
    def hashing_euclid(self, curr_order, model_idx, num_bucket, batch_size):
        num_idx = model_idx.shape[0]
        if curr_order == 0: slice_size = self.input_mat.src_ncol
        else: slice_size = self.input_mat.src_nrow
        curr_line = 5 * (np.random.rand(slice_size) - 0.5)
        curr_line = curr_line / np.linalg.norm(curr_line)
        curr_lines = np.tile(curr_line, num_idx)
        proj_pts = [0 for _ in range(num_idx)]
        
        # Build repeated long vector for the current line
        slices = self.input_mat.extract_slice(curr_order, self.perm_list[curr_order][model_idx[0]].item())         
        for i in range(1, num_idx):
            curr_slice = self.input_mat.extract_slice(curr_order, self.perm_list[curr_order][model_idx[i]].item())            
            slices = np.concatenate((slices, curr_slice))
        
        proj_pts = torch.zeros(num_idx, dtype=torch.double).to(self.i_device)
        with torch.no_grad():
            for i in range(0, num_idx * slice_size, batch_size):
                if num_idx*slice_size - i < batch_size: curr_batch_size = num_idx*slice_size - i
                else: curr_batch_size = batch_size
                temp_vec1 = torch.tensor(curr_lines[i:i+curr_batch_size]).to(self.i_device)
                temp_vec2 = torch.tensor(slices[i:i+curr_batch_size]).to(self.i_device)
                dot_prod = temp_vec1 * temp_vec2
                curr_idx = torch.arange(i,i+curr_batch_size, device=self.i_device)
                
                #print(proj_pts.dtype)
                #print(dot_prod.dtype)
                proj_pts.scatter_(0, curr_idx//slice_size, dot_prod, reduce='add')
        
        proj_pts = proj_pts.cpu().numpy()
        min_point, max_point = min(proj_pts), max(proj_pts)
        seg_len = (max_point - min_point) / (num_bucket - 1)
        start_point = random.uniform(min_point - seg_len, min_point)
        bucket_idx = proj_pts.copy()
        for i in range(num_idx):
            bucket_idx[i] = int((proj_pts[i] - start_point) // seg_len)
            
        return bucket_idx.astype(np.int)
    
    '''
        Use euclid hashing 
        curr_order: 0 for row and 1 for col
    '''
    def change_permutation(self, batch_size, curr_order):
        # Hashing
        _matrix = self.input_mat
        if curr_order == 0: curr_dim = _matrix.src_nrow
        else: curr_dim = _matrix.src_ncol
        
        num_pair = curr_dim//2
        _temp = (curr_dim-2)//2 + 1
        model_idx = 2*np.arange(_temp) + np.random.randint(2, size=_temp)
        num_bucket = curr_dim // 8
        bucket_idx = self.hashing_euclid(curr_order, model_idx, num_bucket, batch_size)
                
        # Build bucket
        buckets = [[] for _ in range(num_bucket)]
        #print(model_idx.size)
        #print(bucket_idx.size)
        #print(num_pair)
        for i in range(num_pair):            
            buckets[bucket_idx[i]].append(model_idx[i])
        if curr_dim % 2 == 1: remains = [curr_dim - 1]
        else: remains = []
            
        # Build pairs within buckets
        first_elem, second_elem = [], []
        for i in range(num_bucket):
            random.shuffle(buckets[i])
            if len(buckets[i]) % 2 == 1:
                rem_part = buckets[i].pop(-1)
                remains.append(rem_part)
                remains.append(rem_part^1)
            
            first_elem = first_elem + buckets[i]
            second_elem_temp = [0 for _ in range(len(buckets[i]))]
            second_elem_temp[0::2] = [elem^1 for elem in buckets[i][1::2]]
            second_elem_temp[1::2] = [elem^1 for elem in buckets[i][0::2]]
            second_elem = second_elem + second_elem_temp
        
        # Build pairs within remains
        random.shuffle(remains)
        if len(remains) % 2 == 1: remains.pop(-1)
        first_elem = first_elem + remains[0::2]
        second_elem = second_elem + remains[1::2]
        first_elem, second_elem = torch.tensor(first_elem, dtype=torch.long, device=self.i_device), torch.tensor(second_elem, dtype=torch.long, device=self.i_device) 
        
        # Initialize variables
        if curr_order == 0: num_slice_entry = _matrix.src_ncol
        else: num_slice_entry = _matrix.src_nrow
        loss_list = torch.zeros(num_pair, device=self.i_device, dtype=torch.double)
       
        # Compute the loss change
        self.model.eval()
        num_total_entry = num_pair * num_slice_entry
        delta_loss, curr_idx = 0., 0
        with torch.no_grad():
            while curr_idx < num_total_entry:
                if batch_size > num_total_entry - curr_idx: curr_batch_size = num_total_entry - curr_idx
                else: curr_batch_size = batch_size
                
                # Get the index of pairs                
                pair_idx = torch.arange(curr_idx, curr_idx + curr_batch_size, dtype=torch.long, device=self.i_device)
                pair_idx = pair_idx // num_slice_entry
                pair_start_idx = curr_idx // num_slice_entry
                pair_end_idx = (curr_idx + curr_batch_size) // num_slice_entry
                if (curr_idx + curr_batch_size) % num_slice_entry != 0: 
                    pair_end_idx += 1
                    
                # Initialize preds and vals
                vals0 = torch.empty(curr_batch_size, dtype=torch.double, device=self.i_device)
                vals1 = torch.empty(curr_batch_size, dtype=torch.double, device=self.i_device)  
                rows0, cols0 = torch.zeros(curr_batch_size, dtype=torch.long, device=self.i_device), torch.zeros(curr_batch_size, dtype=torch.long, device=self.i_device)
                rows1, cols1 = torch.zeros(curr_batch_size, dtype=torch.long, device=self.i_device), torch.zeros(curr_batch_size, dtype=torch.long, device=self.i_device) 
                batch_idx, chunk_size = 0, 0
                for i in range(pair_start_idx, pair_end_idx):
                    # Extract value
                    curr_vals0 = _matrix.extract_slice(curr_order, self.perm_list[curr_order][first_elem[i]].item())
                    curr_vals1 = _matrix.extract_slice(curr_order, self.perm_list[curr_order][second_elem[i]].item())
                    curr_vals0, curr_vals1 = torch.tensor(curr_vals0, device=self.i_device), torch.tensor(curr_vals1, device=self.i_device)
                    if i == pair_start_idx: curr_start_idx = curr_idx % num_slice_entry
                    else: curr_start_idx = 0
                    if i == pair_end_idx - 1:
                        curr_end_idx = (curr_idx + curr_batch_size) % num_slice_entry
                        if curr_end_idx == 0: curr_end_idx = num_slice_entry
                    else: curr_end_idx = num_slice_entry
                    chunk_size = curr_end_idx - curr_start_idx
                    
                    vals0[batch_idx: batch_idx + chunk_size] = curr_vals0[curr_start_idx:curr_end_idx]
                    vals1[batch_idx: batch_idx + chunk_size] = curr_vals1[curr_start_idx:curr_end_idx]
                    
                    # Extract Row and column indices
                    _temp = torch.arange(curr_start_idx, curr_end_idx, dtype=torch.long, device=self.i_device)
                    if curr_order == 0:
                        rows0[batch_idx: batch_idx+chunk_size] = first_elem[i]
                        rows1[batch_idx: batch_idx+chunk_size] = second_elem[i]
                                                
                        cols0[batch_idx: batch_idx+chunk_size] = self.inv_perm_list[1][_temp]
                        cols1[batch_idx: batch_idx+chunk_size] = self.inv_perm_list[1][_temp]
                    else:
                        rows0[batch_idx: batch_idx+chunk_size] = self.inv_perm_list[0][_temp]
                        rows1[batch_idx: batch_idx+chunk_size] = self.inv_perm_list[0][_temp]
                        
                        cols0[batch_idx: batch_idx+chunk_size] = first_elem[i]
                        cols1[batch_idx: batch_idx+chunk_size] = second_elem[i]
                    
                    batch_idx += chunk_size
                
                # Build inputs
                rows0 = rows0.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
                cols0 = cols0.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k                
                input0 = rows0 * self.n_list + cols0 + self._add
                output0 = self.model(input0)
                
                rows1 = rows1.unsqueeze(-1) // self.row_bases % self.m_list  # batch size x self.k
                cols1 = cols1.unsqueeze(-1) // self.col_bases % self.n_list   # batch size x self.k                
                input1 = rows1 * self.n_list + cols1 + self._add
                output1 = self.model(input1)
                loss_list.scatter_(0, pair_idx, -torch.square(output0-vals0)-torch.square(output1-vals1), reduce='add')
                loss_list.scatter_(0, pair_idx, torch.square(output0-vals1)+torch.square(output1-vals0), reduce='add')
                
                curr_idx += curr_batch_size
        
            target_pair = loss_list < 0
            delta_loss = loss_list[target_pair].sum().item()
            first_model_idx, second_model_idx = first_elem[target_pair].clone(), second_elem[target_pair].clone()
            first_mat_idx, second_mat_idx = self.perm_list[curr_order][first_model_idx].clone(), self.perm_list[curr_order][second_model_idx].clone()
            
            self.perm_list[curr_order][first_model_idx], self.perm_list[curr_order][second_model_idx] = second_mat_idx, first_mat_idx
            self.inv_perm_list[curr_order][first_mat_idx], self.inv_perm_list[curr_order][second_mat_idx] = second_model_idx, first_model_idx
            return delta_loss