import numpy as np
import pandas as pd
import math
from keras._tf_keras.keras.layers import Input, Dense, ELU
from keras._tf_keras.keras.models import Model

num = 1
path=f'D:/code/data/data/pair/{num}/'
path2=f'D:/code/tang/dataset{num}/emd/'


with open(path +'clist.txt',encoding='utf-8')as f:
    lines=pd.read_csv(f,header=None)
    clist=lines.iloc[:,0].values.tolist()
    clist = sorted(list(set(clist)))
    cnum = len(clist)

with open(path + 'dlist.txt', encoding='utf-8') as f:
    lines = pd.read_csv(f, header=None)
    dlist = lines.iloc[:, 0].values.tolist()
    dlist = sorted(list(set(dlist)))
    dnum = len(dlist)
print('cnum:',cnum)
print('dnum:',dnum)


def gips(matrix,name):
    amatrix = matrix
    amatrix = amatrix.T
    dgs = np.zeros([dnum, dnum])
    width = 0
    for m in range(dnum):
        width += np.sum(amatrix[m] ** 2) ** 0.5

    for m in range(dnum):
        for n in range(dnum):
            dgs[m, n] = math.exp((np.sum((amatrix[m] - amatrix[n]) ** 2) ** 0.5 * width / dnum) * (-1)) 
            if dgs[m, n] == 1 and m != n:
                dgs[m, n] = 0.9
            if dgs[m, n] <0.005:
                dgs[m, n] = 0

    result = pd.DataFrame(dgs)
    result.to_csv(path2+name+'d_gipsim.csv')

    amatrix = amatrix.T
    rgs = np.zeros([cnum, cnum])

    for m in range(cnum):
        width += np.sum(amatrix[m] ** 2) ** 0.5

    for m in range(cnum):
        for n in range(cnum):
            rgs[m, n] = math.exp((np.sum((amatrix[m] - amatrix[n]) ** 2) ** 0.5 * width / cnum) * (-1))
            if rgs[m, n] == 1 and m != n:
                rgs[m, n] = 0.9
            if rgs[m, n] <0.005:
                rgs[m, n] = 0


    result = pd.DataFrame(rgs)
    result.to_csv(path2 +name+'c_gipsim.csv')


def mesh_similarity(name):
    similarity1 = pd.read_csv(path + 'meshsim1.csv', header=0, encoding='gb18030').values
    similarity2 = pd.read_csv(path + 'meshsim2.csv', header=0, encoding='gb18030').values
    similarity1 = np.array(similarity1)
    similarity2 = np.array(similarity2)
    similarity1 = np.delete(similarity1, 0, axis=1)
    similarity2 = np.delete(similarity2, 0, axis=1)

    dgs = pd.read_csv(path2 +name+'d_gipsim.csv', encoding='gb18030').values
    dgs = np.array(dgs)
    dgs = np.delete(dgs,0,axis=1)

    meshdisname = pd.read_csv(path + 'Mesh_disease.csv', header=0, encoding='gb18030')
    for col in meshdisname.select_dtypes(include=[object]):  
        meshdisname[col] = meshdisname[col].str.lower()

    ds = dgs
    count=0
    for m in range(len(dlist)):
        for n in range(len(dlist)):
            mesh_m = meshdisname[(meshdisname.C1 == str(dlist[m]))].index.tolist()
            mesh_n = meshdisname[(meshdisname.C1 == str(dlist[n]))].index.tolist()
            if mesh_m and mesh_n:
                ds[m, n] = (similarity1[mesh_m[0], mesh_n[0]] + similarity2[mesh_m[0], mesh_n[0]])*0.5
                count+=1
            if m == n:
                ds[m, n] = 1
            if ds[m,n] <=0.001 :
                ds[m,n] = 0

    print(f'count:{count}')
    result = pd.DataFrame(ds)
    result.to_csv(path2+name+'d_descriptor.csv')


def func_similarity(matrix,name):
    am=matrix
    am=np.array(am)

    ds=pd.read_csv(path2+name+'d_descriptor.csv', header=0, encoding='gb18030').values
    ds=np.array(ds)
    ds=np.delete(ds,0,axis=1)

    n_circ=cnum

    cfs = np.zeros((n_circ, n_circ), dtype=np.float32)
    for i in range(n_circ):
        idx = np.nonzero(am[i, :])[0]
        if idx.size == 0:
            continue
        for j in range(i):
            idy = np.nonzero(am[j, :])[0]
            if idy.size == 0:
                continue
            sum1, sum2 = 0, 0
            for k1 in range(len(idx)):
                sum1 = sum1 + max(ds[idx[k1], idy])
            for k2 in range(len(idy)):
                sum2 = sum2 + max(ds[idx, idy[k2]])
            cfs[i, j] = (sum1 + sum2) / (len(idx) + len(idy))
            cfs[j, i] = cfs[i, j]
            if cfs[i,j]==1 and i!=j:
                cfs[i,j]=0.9
                cfs[j,i]=0.9

        for k in range(n_circ):
            cfs[k, k] = 1

    result = pd.DataFrame(cfs)
    result.to_csv(path2+name+'c_funcsim.csv')


def c_descriptor(name):
    cgs = pd.read_csv(path2+name+'c_gipsim.csv',sep=',',header=0, encoding='ISO-8859-1').values
    cgs = np.array(cgs)
    cgs = np.delete(cgs, 0, axis=1)

    cfs=pd.read_csv(path2+name+'c_funcsim.csv', header=0, encoding='ISO-8859-1').values
    cfs = np.array(cfs)
    cfs = np.delete(cfs, 0, axis=1)

    ds = cgs
    for m in range(len(clist)):
        if m%100==0:print(m)
        for n in range(len(clist)):
            if ds[m,n] <= 0.001:
                ds[m,n] = 0
            if cfs[m,n] !=0 :
                ds[m,n] = (ds[m,n] + cfs[m,n])*0.5

    result = pd.DataFrame(ds)
    result.to_csv(path2+name+ 'c_descriptor.csv')

def DNN(train, num, dim=128, epochs=1000, batch_size=32):
    input = Input(shape=(num,))
    encoded = Dense(512, activation='elu')(input)  
    encoded = Dense(256, activation='elu')(encoded)  
    output = Dense(dim)(encoded)  


    decoded = Dense(256, activation='elu')(output)
    decoded = Dense(512, activation='elu')(decoded)
    decoded = Dense(num, activation='tanh')(decoded)

    autoencoder = Model(inputs=input, outputs=decoded)

    encoder = Model(inputs=input, outputs=output)

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train, train, epochs=epochs, batch_size=batch_size, shuffle=True)  
    embed = encoder.predict(train)

    return embed

def DNNsim(name, name2='256',dim=256):
    print(f'start to calculate {name} DNN similarity')

    cs = pd.read_csv( path2+name+ 'c_descriptor.csv', header=0, encoding='ISO-8859-1').values
    cs = np.delete(cs, 0, axis=1)
    cs = np.array(cs,dtype='float32')

    ds = pd.read_csv( path2+name+ 'd_descriptor.csv', header=0, encoding='ISO-8859-1').values
    ds = np.delete(ds, 0, axis=1)
    ds = np.array(ds,dtype='float32')

    circRNA_Embed = DNN(cs,cnum,encoding_dim=dim)
    disease_Embed = DNN(ds,dnum,encoding_dim=dim)

    cd_Embed = np.vstack((circRNA_Embed, disease_Embed))

    cdresult = pd.DataFrame(cd_Embed)
    cdresult.to_csv(path2 + name + '_emd'+name2+'.csv')

