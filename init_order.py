import sys
import numpy as np
import torch
import time
import argparse


def loss_fn(z, batch_size=-1):
    rets = []
    for j in range(0, z.shape[0]-1, batch_size):
        bsize = min(batch_size, z.shape[0]-1-j)
        rets.append((z[j:j+bsize] - z[j+1:j+bsize+1]).pow(2).sum(-1).pow(0.5))
    return torch.cat(rets, dim=-1)


def in_order_search(_tree_dict, curr_node):
    return_list = [curr_node]
    if curr_node not in _tree_dict: return return_list
    for child in _tree_dict[curr_node]:
        return_list = return_list + in_order_search(_tree_dict, child)
    return return_list


def reorder(input_tensor):
    with torch.no_grad():    
        input_tensor = torch.from_numpy(input_tensor)
        dim = len(input_tensor.shape)
        mxsz = (1 << 25)
        change_order, final_orders = [], []
        for i in range(dim):
            mat = input_tensor.to(0).permute(i, *[j for j in range(dim) if j != i]).contiguous().view(input_tensor.shape[i], -1)
            batch_size = (mxsz // mat.shape[-1])           

            adj = torch.zeros(mat.shape[0] * mat.shape[0]).to(0).double()
            for j in range(0, mat.shape[0] * mat.shape[0], batch_size):
                bsize = min(batch_size, mat.shape[0] * mat.shape[0] - j)
                _idx = j + torch.arange(bsize, dtype=torch.long)
                _from, _to = mat[_idx // mat.shape[0]], mat[_idx % mat.shape[0]]
                adj[_idx] = (_from - _to).pow(2).sum(-1).pow(0.5)

            inf = adj.max() * 2 + 1
            adj = adj.view(mat.shape[0], mat.shape[0])
            dist = torch.ones(mat.shape[0], dtype=torch.double).to(0) * inf
            argdist = -torch.ones(mat.shape[0], dtype=torch.long).to(0)
            dist[0] = 0
            mask = torch.zeros(mat.shape[0], dtype=torch.long).to(0)
            edges = []
            mst_size = 0.
            tree_dict = {}
            for j in range(mat.shape[0]):
                new_node = torch.argmin(dist + (mask * inf * 2), dim=-1).item()
                parent_node = argdist[new_node].item()
                if parent_node >= 0:
                    edges.append((new_node, parent_node))
                    # mst_size += adj[new_node, parent_node].item()
                    if parent_node not in tree_dict:
                        tree_dict[parent_node] = []
                    tree_dict[parent_node].append(new_node)

                mask[new_node] += 1
                new_vals = adj[new_node]
                need_update = (dist > new_vals)
                dist[need_update] = new_vals[need_update]
                argdist[need_update] = new_node

            prev_length_sum = loss_fn(mat, batch_size=batch_size).sum().item()
            y = in_order_search(tree_dict, 0)
            y.append(0)
            new_mat = mat[y]
            lengths = loss_fn(new_mat, batch_size=batch_size)
            cut_idx = lengths.argmax(-1)
            final_orders.append(y[cut_idx+1:-1] + y[:cut_idx+1])
            #counts = torch.bincount(torch.LongTensor(final_orders[-1]))
            new_length_sum = loss_fn(mat[final_orders[-1]], batch_size=batch_size).sum().item()
            change_order.append(((prev_length_sum - new_length_sum) > 0.1*prev_length_sum))
            print(f'order: {i}, loss before: {prev_length_sum}, loss after: {new_length_sum}')
            del mat
    
    return change_order, final_orders

# python TensorCodec/init_order.py -lp input/23-NeuTT/action_orig.npy -sp input/23-NeuTT/action.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp', '--load_path', type=str)
    parser.add_argument('-sp', '--save_path', type=str)    
    args = parser.parse_args()
    input_tensor = np.load(args.load_path)
    dim = len(input_tensor.shape)

    start_time = time.time()
    change_order, final_orders = reorder(input_tensor)
    reordered_tensor = input_tensor
    if dim >= 1 and change_order[0]: reordered_tensor = reordered_tensor[final_orders[0]]
    if dim >= 2 and change_order[1]: reordered_tensor = reordered_tensor[:, final_orders[1]]
    if dim >= 3 and change_order[2]: reordered_tensor = reordered_tensor[:, :, final_orders[2]]
    if dim >= 4 and change_order[3]: reordered_tensor = reordered_tensor[:, :, :, final_orders[3]]
    print("Total elapsed time:", time.time() - start_time)
    np.save(args.save_path, reordered_tensor)