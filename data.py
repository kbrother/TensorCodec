import math
import numpy as np
import torch

# Matrix
class _mat:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, input_size, data_name, device):                
        self.src_tensor = np.load(data_name)
        self.src_tensor = self.src_tensor.astype(np.double)
        self.src_dims = list(np.shape(self.src_tensor))                            
        self.order = len(self.src_dims)
        self.dims = [int(np.prod(np.array(input_size[i]))) for i in range(self.order)]
        
        self.src_vals = self.src_tensor.flatten()
        self.real_num_entries = len(self.src_vals)
        self.norm = math.sqrt(np.square(self.src_vals).sum())        
        self.src_base, self._base = [], []
        temp_base = 1
        for i in range(self.order-1, -1, -1):
            self.src_base.insert(0, temp_base)
            temp_base *= self.src_dims[i]
            
        temp_base = 1
        for i in range(self.order-1, -1, -1):
            self._base.insert(0, temp_base)
            temp_base *= self.dims[i]
            
        device = torch.device("cuda:" + str(device))
        self.src_base = torch.tensor(self.src_base, dtype=torch.long, device=device)
        self.src_dims_gpu = torch.tensor(self.src_dims, dtype=torch.long, device=device)        
        self.dims = torch.tensor(self.dims, dtype=torch.long, device=device)        
        self._base = torch.tensor(self._base, dtype=torch.long, device=device)
        print(f'norm of the tensor: {self.norm}')

    def extract_slice(self, curr_order, idx):
        curr_input = [slice(None) for _ in range(self.order)]
        curr_input[curr_order] = idx
        curr_input = tuple(curr_input)
        return self.src_tensor[curr_input].flatten()