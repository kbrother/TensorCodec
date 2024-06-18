import math
import numpy as np
import torch

# Matrix
class _mat:
    '''
        input_size: order x k
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, data_name, device):                
        self.src_tensor = np.load(data_name)
        print(self.src_tensor.shape)
        if self.src_tensor.dtype != "float64":
            self.src_tensor = self.src_tensor.astype(np.float64)
        self.dims = list(np.shape(self.src_tensor))                            
        self.order = len(self.dims)
        self.src_base = []
        temp_base = 1
        for i in range(self.order-1, -1, -1):
            self.src_base.insert(0, temp_base)
            temp_base *= self.dims[i]        
        
        self.vals = self.src_tensor.flatten()    
        self.num_entries = len(self.vals)        
        self.norm = math.sqrt(np.square(self.vals).sum())                                    
            
        device = torch.device("cuda:" + str(device))
        self.src_base = torch.tensor(self.src_base, dtype=torch.long, device=device)
        self.dims_gpu = torch.tensor(self.dims, dtype=torch.long, device=device)                
        print(f'norm of the tensor: {self.norm}')