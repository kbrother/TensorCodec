import numpy as np
import argparse
import torch
from tqdm import tqdm
import math
import sys
from model import TensorCodec
from data import _mat
import copy
import time

def test_perm(n_model, args):
    with torch.no_grad():
        n_model.model.eval()
        curr_loss = n_model.L2_loss(False, args.batch_size)
        print(f'initial loss: {curr_loss}')        
        for i in range(1):
            for j in range(4):
                delta_loss = n_model.change_permutation(args.batch_size, j)
                curr_loss += delta_loss
                print(delta_loss)
                
        print(f'our loss: {curr_loss}, real loss:{n_model.L2_loss(False, args.batch_size)}')
        
def train_model(n_model, args):
    device = torch.device("cuda:" + str(args.device[0]))   
    max_fit = -sys.float_info.max
    prev_fit = -1
    n_model.model.train()
    minibatch_size = n_model.input_mat.real_num_entries // args.num_batch
    
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'compressed size: {n_model.comp_size} bytes\n')    
        
    tol_count = 0
    start_time = time.time()
    for epoch in range(args.epoch): 
        optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr/args.num_batch) 
        n_model.model.train()               
        curr_order = np.random.permutation(n_model.input_mat.real_num_entries)            
        for i in tqdm(range(0, n_model.input_mat.real_num_entries, minibatch_size)):
            if n_model.input_mat.real_num_entries - i < minibatch_size: 
                curr_batch_size = n_model.input_mat.real_num_entries - i
            else: curr_batch_size = minibatch_size
            samples = curr_order[i:i+curr_batch_size]
            
            optimizer.zero_grad()
            mini_loss, mini_norm = n_model.L2_minibatch_loss(True, args.batch_size, samples)
            optimizer.step() 
        
        n_model.model.eval()
        for _dim in range(n_model.order):
            n_model.change_permutation(args.batch_size, _dim)
                        
        with torch.no_grad():
            n_model.model.eval()
            curr_loss = n_model.L2_loss(False, args.batch_size)
            curr_fit = 1 - math.sqrt(curr_loss)/n_model.input_mat.norm
            
            if prev_fit + 1e-4 <= curr_fit: 
                tol_count = 0
                prev_fit = curr_fit
            else: tol_count += 1
                
            if max_fit < curr_fit:
                max_fit = curr_fit                
                prev_model = copy.deepcopy(n_model.model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': prev_model,                    
                    'loss': curr_fit,
                    'perm': n_model.perm_list
                }, args.save_path + ".pt")
            
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch:{epoch}, current fitness: {curr_fit}\n')    
            print(f'epoch:{epoch}, current fitness: {curr_fit}\n')                        
        if tol_count >= args.tol: break
    
    end_time = time.time()
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'running time: {end_time - start_time}\n')    
    print(f'running time: {end_time - start_time}')
    
def test(n_model, args):
    _device = torch.device("cuda:" + str(args.device[0]))     
    checkpoint = torch.load(args.load_path , map_location = _device)
    n_model.model.load_state_dict(checkpoint['model_state_dict'])          
    n_model.perm_list = checkpoint['perm']     
    for i in range(n_model.order):
        n_model.inv_perm_list[i][n_model.perm_list[i]] = torch.arange(n_model.input_mat.dims[i], device=_device)
    
    n_model.model.eval()
    with torch.no_grad():
        curr_loss = n_model.L2_loss(False, args.batch_size)
        print(f"saved loss: {checkpoint['loss']}, computed loss: {1 - math.sqrt(curr_loss) / n_model.input_mat.norm}")
    
            
# python TensorCodec/main.py train -d uber -de 0 1 2 3 -rk 7 -hs 9 -sp output/uber1_r7_h9 -e 5000 -lr 1e-1 -m gru -nb 100 -t 100 -b 8388608
# python main.py check_sum -d uber -de 0 1 2 3 -rk 5 -hs 10 
# python main.py test -d action -de 0 1 2 3 -rk 6 -hs 8 
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train')
    parser.add_argument("-d", "--dataset", type=str)   
            
    parser.add_argument(
        "-de", "--device",
        action="store", nargs='+', type=int
    )    
    
    '''
    parser.add_argument(
        "-p", "--perms",
        action="store", default=[], nargs='+', type=int
    )  
    '''
    
    parser.add_argument(
        "-rk", "--rank",
        action="store", default=12, type=int
    )
    
    parser.add_argument(
        "-lr", "--lr",
        action="store", default=1e-3, type=float
    )
    
    parser.add_argument(
        "-e", "--epoch",
        action="store", default=2000, type=int
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        action="store", default=2**22, type=int
    )
    
    parser.add_argument(
        "-nb", "--num_batch",
        action="store", default=20, type=int
    )
    
    parser.add_argument(
        "-sp", "--save_path",
        action="store", default="./params/", type=str
    )
    
    parser.add_argument(
        "-lp", "--load_path",
        action="store", default="./params/", type=str
    )
        
    parser.add_argument(
        "-hs", "--hidden_size",
        action="store", default=11, type=int
    )
    
    parser.add_argument(
        "-t", "--tol", 
        action="store", default=10, type=int
    )
    
    parser.add_argument(
        "-m", "--model",
        action="store", default="lstm", type=str
    )
    
    args = parser.parse_args()      
    # decompsress m_list and n_list
    with open("TensorCodec/input_size/" + args.dataset + ".txt") as f:
        lines = f.read().split("\n")
        input_size = [[int(word) for word in line.split()] for line in lines if line]        
     
    input_mat = _mat(input_size, "input/" + args.dataset + ".npy", args.device[0])        
    #input_mat = _mat(input_size, "input/23-NeuTT/" + args.dataset + ".npy", args.device[0])        
    #input_mat = _mat(input_size, "data/" + args.dataset + ".npy", args.device[0])        
    print("load finish")
    if args.action == "train":
        n_model = TensorCodec(input_mat, args.rank, input_size, args.hidden_size, args.device, args.model)
        train_model(n_model, args)
    elif args.action == "test":
        n_model = TensorCodec(input_mat, args.rank, input_size, args.hidden_size, args.device, args.model)
        test(n_model, args)        
    elif args.action == "test_perm":
        n_model = TensorCodec(input_mat, args.rank, input_size, args.hidden_size, args.device, args.model)      
        test_perm(n_model, args)      
    elif args.action == "check_sum":
        n_model = TensorCodec(input_mat, args.rank, input_size, args.hidden_size, args.device, args.model)        
        k = len(input_size[0])
        first_mat = np.ones((1, args.rank))
        middle_mat = np.ones((args.rank, args.rank))
        final_mat = np.ones((args.rank, 1))
        
        gt_result = first_mat
        for i in range(k-2):
            gt_result = np.matmul(gt_result, middle_mat)
        gt_result = np.matmul(gt_result, final_mat)
        print(f'model sum: {n_model.entry_sum(args.batch_size)}, gt result: {gt_result.item()}')
    else:
        assert(False)
