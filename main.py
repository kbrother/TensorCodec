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

                
def train_model(n_model, args):
    device = torch.device("cuda:" + str(args.device[0]))   
    max_fit = -sys.float_info.max
    prev_fit = -1
    n_model.model.train()
    minibatch_size = n_model.input_mat.num_entries // args.num_batch
    
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'compressed size: {n_model.comp_size} bytes\n')    
        
    tol_count = 0
    start_time = time.time()
    for epoch in range(args.epoch): 
        optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr/args.num_batch) 
        n_model.model.train()               
        curr_order = np.random.permutation(n_model.input_mat.num_entries)            
        for i in tqdm(range(0, n_model.input_mat.num_entries, minibatch_size)):
            if n_model.input_mat.num_entries - i < minibatch_size: 
                curr_batch_size = n_model.input_mat.num_entries - i
            else: curr_batch_size = minibatch_size
            samples = curr_order[i:i+curr_batch_size]
            
            optimizer.zero_grad()
            n_model.L2_minibatch_loss(args.batch_size, samples)
            optimizer.step() 
        
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
                }, args.save_path + ".pt")
            
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch:{epoch}, current fitness: {curr_fit}\n')    
            print(f'epoch:{epoch}, current fitness: {curr_fit}\n')                        
        if tol_count >= args.tol: break
    
    end_time = time.time()
    with open(args.save_path + ".txt", 'a') as lossfile:
        lossfile.write(f'running time: {end_time - start_time}\n')    
    print(f'running time: {end_time - start_time}')
    

# python TensorCodec/main.py train -d action -de 0 1 2 3 -rk 5 -hs 5 -sp output/action_r5_h5 -e 1000 -lr 1e-1 -t 100 -b 8388608
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
        action="store", default=100, type=int
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

    args = parser.parse_args()           
    input_mat = _mat("data/23-NeuTT/" + args.dataset + ".npy", args.device[0])        
    #input_mat = _mat(input_size, "input/23-NeuTT/" + args.dataset + ".npy", args.device[0])        
    #input_mat = _mat(input_size, "data/" + args.dataset + ".npy", args.device[0])        
    print("load finish")
    if args.action == "train":
        n_model = TensorCodec(input_mat, args.rank, args.hidden_size, args.device)
        train_model(n_model, args)
    else:
        assert(False)
