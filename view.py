import torch
import pandas as pd
import networkx as nx
import numpy as np

path='D:/code/data/data/pair/2/'
path2='D:/code/tang/dataset2/view/'
# all_list:     c-d_pair:     c-m_pair:     m-l_pair:     m-d_pair:     l-d_pair:
# ccc  c1        c1  d1        c1  m1        m1  l1        m1  d1        l1  d1
# ddd  d2        c2  d2        c2  m2        m2  l2        m2  d2        l2  d2

all=pd.read_csv(path +'all_list.txt',sep='\t',header=None,encoding='utf-8')
cd=pd.read_csv(path +'c-d_pair.txt',sep='\t',header=None)
cm=pd.read_csv(path +'c-m_pair.txt',sep='\t',header=None)
ml=pd.read_csv(path +'m-l_pair.txt',sep='\t',header=None)
md=pd.read_csv(path +'m-d_pair.txt',sep='\t',header=None)
ld=pd.read_csv(path +'l-d_pair.txt',sep='\t',header=None)
cnum = len(all[all[1].str.startswith('c')])
dnum = len(all[all[1].str.startswith('d')])
mnum = len(all[all[1].str.startswith('m')])
lnum = len(all[all[1].str.startswith('l')])
with open(path + 'num.txt', 'w') as f:
    f.write(f'cnum: {cnum}\n')
    f.write(f'dnum: {dnum}\n')
    f.write(f'mnum: {mnum}\n')
    f.write(f'lnum: {lnum}\n')
    
#%%
clist=[pair for pair in cd.iloc[:,0]]
clist=list(set(clist))
clist.sort(key=lambda x: int(x[1:]))

dlist=[pair for pair in cd.iloc[:,1]]
dlist=list(set(dlist))
dlist.sort(key=lambda x: int(x[1:]))

mlist1=[pair for pair in cm.iloc[:,1]]
mlist2=[pair for pair in md.iloc[:,0]]
mlist3=[pair for pair in ml.iloc[:,0]]
mlist=mlist1+mlist2+mlist3
mlist=list(set(mlist))
mlist.sort(key=lambda x: int(x[1:]))

llist1=[pair for pair in ml.iloc[:,1]]
llist2=[pair for pair in ld.iloc[:,0]]
llist1=sorted(list(set(llist1)))
llist2=sorted(list(set(llist2)))
llist=llist1+llist2
llist=list(set(llist))
llist.sort(key=lambda x: int(x[1:]))

hetero_graph=nx.Graph()
for i in range(0, cnum):
    hetero_graph.add_node(all.iloc[i][1], type='c')
for i in range(cnum, cnum+dnum):
    hetero_graph.add_node(all.iloc[i][1], type='d')
for i in range(cnum + dnum, cnum + dnum + mnum):
    hetero_graph.add_node(all.iloc[i][1], type='m')
for i in range(cnum + dnum + mnum, cnum + dnum + mnum + lnum):
    hetero_graph.add_node(all.iloc[i][1], type='l')

def add_edges(edges, graph):

    for index, row in edges.iterrows():
        graph.add_edge(f'{row[0]}', f'{row[1]}')
        graph.add_edge(f'{row[1]}', f'{row[0]}')

    return graph

hetero_graph = add_edges(cd, hetero_graph)
hetero_graph = add_edges(cm, hetero_graph)
hetero_graph = add_edges(ml, hetero_graph)
hetero_graph = add_edges(md, hetero_graph)
hetero_graph = add_edges(ld, hetero_graph)

def build_adjacency_matrix(df, nodes_list1, nodes_list2):
    adjacency_matrix = np.zeros((len(nodes_list1), len(nodes_list2)))

    for _, row in df.iterrows():
        indx1, indx2 = int(row[0][1:]), int(row[1][1:])
        adjacency_matrix[indx1, indx2] = 1

    return adjacency_matrix

cd_adj = build_adjacency_matrix(cd, clist, dlist)
cm_adj = build_adjacency_matrix(cm, clist, mlist)
ml_adj = build_adjacency_matrix(ml, mlist, llist)
md_adj = build_adjacency_matrix(md, mlist, dlist)
ld_adj = build_adjacency_matrix(ld, llist, dlist)


#%%
cmd_weight =np.matmul(cm_adj, md_adj)
cmld_weight = np.matmul(np.matmul(cm_adj, ml_adj), ld_adj)
cmcd_weight = np.matmul(np.matmul(cm_adj, cm_adj.T), cd_adj)
cdmd_weight = np.matmul(cd_adj, np.matmul(md_adj.T, md_adj))
cdld_weight = np.matmul(cd_adj, np.matmul(ld_adj.T, ld_adj))
cdcd_weight = np.matmul(np.matmul(cd_adj, cd_adj.T), cd_adj)

cmlmd_weight = np.matmul(np.matmul(cm_adj, ml_adj), np.matmul(ml_adj.T, md_adj))
cdlmd_weight = np.matmul(cd_adj, np.matmul(ld_adj.T, np.matmul(ml_adj.T, md_adj)))
cdmld_weight = np.matmul(cd_adj, np.matmul(md_adj.T, np.matmul(ml_adj, ld_adj)))


#%% 
def compute_DPRel(weight):
    
    c_degrees = dict(hetero_graph.degree([node for node in hetero_graph.nodes if node.startswith('c')]))
    d_degrees = dict(hetero_graph.degree([node for node in hetero_graph.nodes if node.startswith('d')]))
    DPRel_matrix = np.zeros((cnum, dnum))
    all_DPRel = set()
    for c_node in range(cnum):
        for d_node in range(0, dnum):
            c_d_degree=np.add(c_degrees[f'c{c_node}'], d_degrees[f'd{d_node}'])
            weight_sum1=np.sum(weight, axis=1)
            weight_sum0=np.sum(weight, axis=0)
            DPRel=np.divide(np.dot(weight[c_node,d_node], c_d_degree),
                            np.add(np.dot(d_degrees[f'd{d_node}'], weight_sum1[c_node]),
                                   np.dot(c_degrees[f'c{c_node}'], weight_sum0[d_node])))
            if weight[c_node,d_node] == 0:
                DPRel = 0
            if weight_sum1[c_node] == 0 and weight_sum0[d_node] == 0:
                DPRel = 0
            all_DPRel.add((f'c{c_node}', f'd{d_node}', DPRel))
            DPRel_matrix[c_node, d_node] = DPRel


    return all_DPRel, DPRel_matrix

cd_DPRel, cd_DPRel_matrix = compute_DPRel(cd_adj)
cmd_DPRel, cmd_DPRel_matrix = compute_DPRel(cmd_weight)
cmld_DPRel, cmld_DPRel_matrix = compute_DPRel(cmld_weight)
cmcd_DPRel, cmcd_DPRel_matrix = compute_DPRel(cmcd_weight)
cdmd_DPRel, cdmd_DPRel_matrix = compute_DPRel(cdmd_weight)
cdld_DPRel, cdld_DPRel_matrix = compute_DPRel(cdld_weight)
cdcd_DPRel, cdcd_DPRel_matrix = compute_DPRel(cdcd_weight)

cmlmd_DPRel, cmlmd_DPRel_matrix = compute_DPRel(cmlmd_weight)
cdlmd_DPRel, cdlmd_DPRel_matrix = compute_DPRel(cdlmd_weight)
cdmld_DPRel, cdmld_DPRel_matrix = compute_DPRel(cdmld_weight)


#%%
def build_view(metapath):
    view=np.zeros((cnum,dnum))
    view2=np.zeros((cnum,dnum))
    view3=np.zeros((cnum,dnum))
    num=0
    for i in metapath:
        idx1, idx2=int(i[0][1:]), int(i[-2][1:])
        view3[idx1, idx2] = i[-1]
        if i[-1]>0.005:
            view[idx1, idx2] = 1
            view2[idx1, idx2] = i[-1]
            num+=1
    print(num/cnum/dnum,"\t",num)
    return view, view2, view3

cd_view=cd_adj

cmd_view,cmd_view2, cmd_view3=build_view(cmd_DPRel)

cmld_view,cmld_view2,cmld_view3=build_view(cmld_DPRel)
cmcd_view,cmcd_view2,cmcd_view3=build_view(cmcd_DPRel)
cdmd_view,cdmd_view2,cdmd_view3=build_view(cdmd_DPRel)
cdld_view,cdld_view2,cdld_view3=build_view(cdld_DPRel)
cdcd_view,cdcd_view2,cdcd_view3=build_view(cdcd_DPRel)

cmlmd_view,cmlmd_view2,cmlmd_view3=build_view(cmlmd_DPRel)
cdlmd_view,cdlmd_view2,cdlmd_view3=build_view(cdlmd_DPRel)
cdmld_view,cdmld_view2,cdmld_view3=build_view(cdmld_DPRel)

#%%
np.save(path2 +'cd_view', cd_view)
np.save(path2 +'cmd_view', cmd_view)
np.save(path2 +'cmld_view', cmld_view)
np.save(path2 +'cmcd_view', cmcd_view)
np.save(path2 +'cdmd_view', cdmd_view)
np.save(path2 +'cdld_view', cdld_view)
np.save(path2 +'cdcd_view', cdcd_view)
np.save(path2 +'cmlmd_view', cmlmd_view)
np.save(path2 +'cdlmd_view', cdlmd_view)
np.save(path2 +'cdmld_view', cdmld_view)
#%%

