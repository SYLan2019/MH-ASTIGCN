#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import time
import argparse
from scipy.optimize import linprog


np.seterr(divide='ignore', invalid='ignore')
myN = 0

def emd(p, q, D):
    '''
    通过线性规划求emd
    p.shape=[m], q.shape=[n], D.shape=[m,n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    '''
    global myN
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = np.array(D)
    D = D.reshape(-1)

    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    myresult = result.fun

    return myresult

def traffic_distance(x, y):
    """
    x.shape=[m,d], y.shape=[n,d]
    """
    x, y = np.array(x), np.array(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return emd(p, q, D)


def temporal_similarity(x, y, normal, transpose):
    """1 - WRD
    x.shape=[m,d], y.shape=[n,d]
    """
    if normal:
        x = normalize(x)
        x = normalize(x)
    if transpose:
        x = np.transpose(x)
        y = np.transpose(y)
    return 1 - traffic_distance(x, y)


def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PEMS04", help="Dataset path.")
parser.add_argument("--period", type=int, default=288, help="Time series perios.")
parser.add_argument("--sparsity", type=float, default=0.01, help="sparsity of spatial graph")

args = parser.parse_args()

df = np.load(args.dataset+'/'+args.dataset+".npz")['data']
num_samples,ndim,_ = df.shape

num_train = int(num_samples * 0.6)
num_tmd=int(num_train/args.period)*args.period
data=df[:num_tmd,:,:1].reshape([-1,args.period,ndim])



d=np.zeros([ndim,ndim])
t0=time.time()
for i in range(ndim):
    t1=time.time()
    for j in range(i+1,ndim):
        d[i, j] = temporal_similarity(data[:, :, i], data[:, :, j], normal=False, transpose=False)
        # print(j)
        print('\r', j, end='', flush=True)
    t2=time.time()
    print('Line',i,'finished in',t2-t1,'seconds.')

tmd=d+d.T


np.save(args.dataset+'/'+args.dataset+"-TMD.npy",tmd)
print("The calculation of time series is done!")
t3=time.time()
print('total finished in',t3-t0,'seconds.')
adj = np.load(args.dataset+'/'+args.dataset+"-TMD.npy")
id_mat = np.identity(ndim)
adjl = adj + id_mat
adjlnormd = adjl/adjl.mean(axis=0)

adj = 1 - adjl + id_mat
w_adj = np.zeros([ndim,ndim])
# w_adj = adj
adj_percent = args.sparsity

top = int(ndim * adj_percent)

for i in range(adj.shape[0]):
    a = adj[i,:].argsort()[0:top]
    for j in range(top):
        w_adj[i, a[j]] = 1


print("Total route number: ", ndim)
print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(ndim*ndim))

pd.DataFrame(w_adj).to_csv(args.dataset+'/'+args.dataset+"-TMD.csv", index = False, header=None)

print("The weighted matrix of temporal graph is generated!")