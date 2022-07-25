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