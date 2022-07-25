import numpy as np
import argparse
import torch
from tqdm import tqdm
import math
import sys
from model import NeuKron_TT, TT
from data import _mat
import copy

def train_model(n_model, args):
    device = torch.device("cuda:" + str(args.device[0]))
    optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr)    
    max_fit = sys.float_info.min
    n_model.model.train()
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        prev_loss = n_model.L2_loss(True, args.batch_size)
        opt_state_dict = copy.deepcopy(optimizer.state_dict())
        prev_model = copy.deepcopy(n_model.model.state_dict())
        optimizer.step()        
        prev_fit = 1 - math.sqrt(prev_loss) / n_model.input_mat.norm   
        
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
        print(f'max fit: {max_fit}')
          
def guide_train(n_model, args, tt_model):
    device = torch.device("cuda:" + str(args.device[0]))
    optimizer = torch.optim.Adam(n_model.model.parameters(), lr=args.lr)    
    max_fit = -sys.float_info.max
    for epoch in range(args.epoch):
        n_model.model.train()
        optimizer.zero_grad()        
        prev_loss = n_model.L2_guide_loss(True, tt_model, args.batch_size)        
        optimizer.step()
        
        n_model.model.eval()
        with torch.no_grad():
            sq_loss = n_model.L2_loss(False, args.batch_size)
            curr_fit = 1 - math.sqrt(sq_loss) / n_model.input_mat.norm
            with open(args.save_path + ".txt", 'a') as lossfile:
                lossfile.write(f'epoch:{epoch}, train loss: {curr_fit}\n')    
                print(f'epoch:{epoch}, train loss: {curr_fit}\n')
        
        if max_fit < curr_fit:
            print("here")
            max_fit = curr_fit
            torch.save({
                'epoch': epoch,
                'model_state_dict': n_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': max_fit
            }, args.save_path + ".pt")
            
            
# python main.py train -d gms5 -m 2 9 -n 2 9 -de 1 2 3 -rk 20 -hs 20 -sp results/gms5/hs20_r20 -e 200
# python main.py guide_train -d gms5 -m 2 9 -n 2 9 -de 0 -rk 10 -hs 10 -sp results/gms5/guide/10_norm -e 2000 -lr 10
# python main.py check_output -d gms5 -m 2 9 -n 2 9 -de 0 -rk 10 -hs 10 -sp results/gms5/guide/10_norm.pt 
# python main.py check_tt -d gms5 -m 2 9 -n 2 9 -de 0 
# python main.py train -d hsi -m 2 10 -n 2 9 3 1 -de 1 2 3 -rk 30 -hs 10 -sp results/hsi/rk40_hs10 -e 200
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='train')
    parser.add_argument("-d", "--dataset", type=str)   
    
    parser.add_argument(
        "-n", "--n_comp",
        action="store", nargs='+', type=int
    )
    
    parser.add_argument(
        "-m", "--m_comp",
        action="store", nargs='+', type=int
    )
    
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
        action="store", default=2**14, type=int
    )
    
    parser.add_argument(
        "-sp", "--save_path",
        action="store", default="./params/", type=str
    )
    
    parser.add_argument(
        "-hs", "--hidden_size",
        action="store", default=11, type=int
    )
    
    args = parser.parse_args()      
    # decompsress m_list and n_list
    _cnt = 0
    m_list = []
    while _cnt < len(args.m_comp):
        m_list = m_list + [args.m_comp[_cnt] for _ in range(args.m_comp[_cnt + 1])]
        _cnt += 2
    _cnt = 0
    n_list = []
    while _cnt < len(args.n_comp):
        n_list = n_list + [args.n_comp[_cnt] for _ in range(args.n_comp[_cnt + 1])]
        _cnt += 2    
                
    num_row, num_col = 1, 1
    for m in m_list: num_row *= m
    for n in n_list: num_col *= n
    input_mat = _mat("../data/" + args.dataset + "_mat.npy", num_row, num_col)
        
    if args.action == "train":
        n_model = NeuKron_TT(input_mat, args.rank, m_list, n_list, args.hidden_size, args.device)
        train_model(n_model, args)
    elif args.action == "check_tt":
        tt_model = TT(input_mat, args.rank, m_list, n_list, args.device[0], args.dataset)       
        print(f'fitness: {tt_model.fitness(args.batch_size)}')
    elif args.action == "guide_train":
        n_model = NeuKron_TT(input_mat, args.rank, m_list, n_list, args.hidden_size, args.device)         
        tt_model = TT(input_mat, args.rank, m_list, n_list, args.device[0], args.dataset)
        print(f'fitness: {tt_model.fitness(args.batch_size)}')
        guide_train(n_model, args, tt_model)
    
    elif args.action == "check_output":
        n_model = NeuKron_TT(input_mat, args.rank, m_list, n_list, args.hidden_size, args.device)         
        tt_model = TT(input_mat, args.rank, m_list, n_list, args.device[0], args.dataset)
        print(f'fitness: {tt_model.fitness(args.batch_size)}')        
        checkpoint = torch.load(args.save_path)
        n_model.model.load_state_dict(checkpoint['model_state_dict'])       
        sq_loss = n_model.L2_loss(False, args.batch_size)
        curr_fit = 1 - math.sqrt(sq_loss) / n_model.input_mat.norm
        print(f'fitness: {curr_fit}')
        n_model.check_output(256, 256, tt_model)        
    else:
        assert(False)