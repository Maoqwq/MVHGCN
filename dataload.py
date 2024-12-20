import random
import numpy as np
import pandas as pd
import torch

def get_neg(all, matrix, cs):
    neg_dict = {}
    for i in range(all.shape[0]):
        pos_num = np.sum(matrix[i] == 1)
        adjn = {(i, k) for k in range(all.shape[1]) if all[i, k] == 0}
        neg = adjn.copy()
        
        for j in range(all.shape[0]):
            if i != j and np.any((all[i] > 0) & (all[j] > 0)):
                adjn -= {(i, index) for index in range(all.shape[1]) if all[j, index] > 0 and all[i, index] == 0}
        try:
            adjn = random.sample(list(adjn), k=int(pos_num))
        except:
            cd = np.dot(cs, matrix)
            idx = np.argsort(cd[i])
            negl = [(i, v) for v in idx]
            negl = [v for v in negl if v not in adjn]
            adjn = list(adjn) + negl[:pos_num - len(adjn)]

        for k, v in adjn:
            neg_dict.setdefault(k, []).append(v)
    
    return neg_dict


def get_pos(matrix):
    pos= []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] == 1:
                pos.append((i,j))
    pos_dict = {}
    for k, v in pos:
        pos_dict.setdefault(k, []).append(v)
    return pos_dict


def dataload(args):
    path = args.data_path
    viewname = args.view
    m = 5
    if viewname is None:
        viewname = ['cd_view']
    A, view_embed = {0: [], 1: [], 2: [], 3: [], 4: []}, {0: None, 1: None, 2: None, 3: None, 4: None}
    pos_sample, neg_sample = {0: None, 1: None, 2: None, 3: None, 4: None}, {0: None, 1: None, 2: None, 3: None, 4: None}

    set = path.split('dataset')[-1][:-1]
    if set=='1':
        numc = 2480
        numd = 101
    elif set=='2':
        numc = 2640
        numd = 181
    elif set=='3':
        numc = 1080
        numd = 172

    indices = np.arange(numc)
    np.random.shuffle(indices)
    group = np.split(indices, m)

    adj_all = 0
    for view in viewname:
        adj_matrix=np.load(path+'view/'+view+'.npy')
        adj_all += adj_matrix

        for i in range(m):
            adj_splot = adj_matrix[group[i]]
            I1, I2 = np.eye(adj_splot.shape[0]), np.eye(adj_splot.shape[1])
            A[i].append(torch.tensor(np.block([[I1, adj_splot], [adj_splot.T, I2]]), dtype=torch.float32))

        if view==viewname[-1]:
            with open(path+f'emd/emd{args.hidden_dim}.csv') as f:
                embed = np.array(pd.read_csv(f, header=0).values)
                embed = np.delete(embed, 0, axis=1)
                u_embed, i_embed = embed[:numc,:], embed[numc:,:]
            with open(path+f'emd/cs.csv') as f:
                cs = np.array(pd.read_csv(f, header=0).values)
                cs = np.delete(cs, 0, axis=1)

            neg_t=get_neg(adj_all,adj_matrix,cs)
            pos_t=get_pos(adj_matrix)
            for i in range(m):
                pos_sample[i] = {idx: pos_t[key] for idx,key in enumerate(group[i])}
                neg_sample[i] = {idx: neg_t[key] for idx,key in enumerate(group[i])}
                view_embed[i] = torch.cat((torch.tensor(u_embed[group[i]], dtype=torch.float32), torch.tensor(i_embed, dtype=torch.float32)), dim=0)

    return A, view_embed,pos_sample,neg_sample,numc//m,numd




