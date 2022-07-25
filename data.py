import math
import numpy as np

# Matrix
class _mat:
    def __init__(self, data_name, num_row, num_col):
        self.num_row, self.num_col = num_row, num_col
        self.src_matrix = np.load(data_name)
        self.src_matrix = self.src_matrix.astype(np.double)
        
        self.src_nrow, self.src_ncol = np.shape(self.src_matrix)[0], np.shape(self.src_matrix)[1]
        self._matrix = self.src_matrix.copy()
        if (num_row > self.src_nrow) or (num_col > self.src_ncol):
            self._matrix = np.pad(self._matrix, ((0,num_row-self.src_nrow), (0,num_col-self.src_ncol)), 'constant')
            
        self.vals = self._matrix.flatten()    
        self.src_vals = self.src_matrix.flatten()
        self.real_num_entries = len(self.src_vals)
        self.num_entries = len(self.vals)
        self.norm = math.sqrt(np.square(self.src_vals).sum())