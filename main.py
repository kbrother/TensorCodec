import numpy as np
import argparse
import torch
from tqdm import tqdm
import math
import sys
from model import NeuKron_TT
from data import _mat
import copy

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
        lossfile.write(f'num params: {n_model.num_params}\n')    
        
    tol_count = 0
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
            lossfile.write(f'epoch:{epoch}, train loss: {curr_fit}\n')    
            print(f'epoch:{epoch}, train loss: {curr_fit}\n')                        
        if tol_count >= 10: break
    
def retrain(n_model, args):
    checkpoint = torch.load(args.load_path)
    n_model.model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda:" + str(args.device[0]))
    optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr)    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    max_fit = -sys.float_info.max
    n_model.model.train()
    for epoch in range(checkpoint['epoch']+1, args.epoch):
        optimizer.zero_grad()
        prev_loss = n_model.L2_loss(True, args.batch_size)
        opt_state_dict = copy.deepcopy(optimizer.state_dict())
        prev_model = copy.deepcopy(n_model.model.state_dict())
        optimizer.step()        
        prev_fit = 1 - math.sqrt(prev_loss)/n_model.input_mat.norm   
        
        with open(args.save_path + ".txt", 'a') as lossfile:
            lossfile.write(f'epoch:{epoch}, train loss: {prev_fit}\n')    
            print(f'epoch:{epoch}, train loss: {prev_fit}\n')
            
        if max_fit < prev_fit:
            max_fit = prev_fit
            torch.save({
                'epoch': epoch,
                'model_state_dict': prev_model,
                'optimizer_state_dict': opt_state_dict,
                'loss': prev_loss
            }, args.save_path + ".pt")            
            
# python 22-TT-train/main.py train -d turb -de 3 -rk 5 -hs 10 -sp output/turb -e 100 -lr 1e-1
# python main.py check_sum -d uber -de 0 1 2 3 -rk 5 -hs 10 
# python main.py test_perm -d absrob -de 0 1 2 3 -rk 5 -hs 10 
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
    
    args = parser.parse_args()      
    # decompsress m_list and n_list
    with open("22-TT-train/input_size/" + args.dataset + ".txt") as f:
        lines = f.read().split("\n")
        input_size = [[int(word) for word in line.split()] for line in lines if line]        
                
    input_mat = _mat(input_size, "data/" + args.dataset + ".npy", args.device[0])        
    print("load finish")
    if args.action == "train":
        n_model = NeuKron_TT(input_mat, args.rank, input_size, args.hidden_size, args.device)
        train_model(n_model, args)
    elif args.action == "retrain":
        n_model = NeuKron_TT(input_mat, args.rank, m_list, n_list, args.hidden_size, args.device)        
        retrain(n_model, args)        
    elif args.action == "test_perm":
        n_model = NeuKron_TT(input_mat, args.rank, input_size, args.hidden_size, args.device)      
        test_perm(n_model, args)      
    elif args.action == "check_sum":
        n_model = NeuKron_TT(input_mat, args.rank, input_size, args.hidden_size, args.device)        
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
