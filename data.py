import math
import numpy as np

# Matrix
class _mat:
    '''
        dims: dimensions with padded size, size is order
    '''
    def __init__(self, data_name, device):                
        self.src_tensor = np.load(data_name)
        self.src_tensor = self.src_tensor.astype(np.double)
        self.src_dims = list(np.shape(self.src_tensor))                
        self.order = len(self.src_dims)
        
        self.src_vals = self.src_tensor.flatten()
        self.real_num_entries = len(self.src_vals)
        self.norm = math.sqrt(np.square(self.src_vals).sum())        
        self.base = []
        temp_base = 1
        for i in range(self.order, -1, -1):
            self.base.insert(0, temp_base)
            temp_base *= self.src_dims[i]
        self.base = torch.LongTensor(temp_base, device=device)
        self.src_dims_gpu = torch.LongTensor(self.src_dims, device=device)
        print(f'norm of the tensor: {self.norm}')

    def extract_slice(self, curr_order, idx):
        curr_input = [slice(None) for _ in range(self.order)]
        curr_input[curr_order] = idx
        curr_input = tuple(curr_input)
        return self.src_matrix[curr_input]