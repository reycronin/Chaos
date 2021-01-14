from mpi4py import MPI
import sys
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

def remainder(OMEGA, K, theta_o, rem, col, start):
    r = np.zeros((rem, col))
    for i in range(rem):
        for j in range(col):
            k = K[start + rem*rank+i][j]
            omega = OMEGA[start + rem*rank+i][j]
            r[i][j] = k
    return r

def getwind(OMEGA, K, W, theta_o):
    col = len(K)
    row = len(K[0])
    num = row//size # number of rows each processor takes 
    w = np.zeros((num, col))
    rem = row - size*num
    r = 0
    if rank == 0 and rem != 0:
        start = row - rem
        r = remainder(OMEGA, K, theta_o, rem, col, start)
        
    for i in range(num):
        for j in range(col):
            k = K[num*rank+i][j]
            omega = OMEGA[num*rank+i][j]
            w[i][j] = k
    return w, r

theta_o = .5
dx = .01
y = np.arange(0,1+dx,dx)
x = np.arange(0,1+dx,dx)
OMEGA, K = np.meshgrid(x,y)
W = OMEGA*0.0

w,r = getwind(OMEGA, K, W, theta_o)

q = comm.gather(w, root = 0)
if rank == 0:
    row = len(K[0])
    num = row//size # number of rows each processor takes
    rem = row - size*num
    for i in range(len(q)):
        for j in range(len(q[0])):
            W[(i*num)+j][:] = q[i][j][:]
    if rem != 0 and isinstance(r,int) != True:
        start = row - rem
        for i in range(rem):
            W[start+i][:] = r[i][:]
            print(start+i)
    
    print(r)
    cc = plt.pcolormesh(OMEGA,K,W, cmap ='gnuplot')
    plt.axis('scaled')
    cbar = plt.colorbar()
    plt.xlabel('$\Omega$')
    plt.ylabel('K')
    cbar.set_label('number of iterations where |$\Theta - \Theta_o$| $< \epsilon$')
    plt.savefig('pic.png')
    plt.show()
    
    mat = np.matrix(W)
    with open('W.txt','wb') as f:
        for line in mat: 
            np.savetxt(f, line, fmt='%.2f')



