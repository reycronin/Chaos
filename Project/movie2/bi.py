from mpi4py import MPI
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import time

t0 = time.time()

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

def circle(theta, omega, K):
    return (theta + omega - K * np.sin(2 * np.pi * theta)) %1

def remainder(omega, K, rem, col,row, start):
    r = np.zeros((row, col))
    for i in range(rem):
        k = K[start+i][0]
        for j in range(col):
            theta = THETA[start+i][j]
            for kk in range(10):
                theta = circle(theta, omega, k)
            index = int(((theta +.5)%1)*col)
            r[i][index] += 1
    return r

def getwind(K, THETA, omega):
    col = len(K[0])
    row = len(K)
    num = row//size # number of rows each processor takes 
    w = np.zeros((row, col))
    rem = row - size*num
    r = 0
    if rank == 0 and rem != 0:
        start = row - rem
        r = remainder(omega, K, rem, col, row, start)
    for i in range(row):
        k = K[i][0]
        for j in range(col):
            theta = THETA[0][j]
            for kk in range(10):
                theta = circle(theta, omega,k)
                index = int(((theta +.5)%1)*col)
            w[i][index] += 1
    return w,r

dx = .001
y = np.arange(0,2+dx,dx)
x = np.arange(0,1+dx,dx)
THETA, K = np.meshgrid(x,y)
for om in range(20):
    omega = om / 20 + .0001
    w,r = getwind(K, THETA,omega)



    if rank == 0 :
        W = K*0.0
    else:
        W =None

    comm.Reduce([w, MPI.DOUBLE],[W, MPI.DOUBLE], op=MPI.SUM, root = 0)

    if rank == 0:
        W = W + r 
        row = len(K)
        col = len(K[0])
        
        for i in range(row):
            for j in range(col):
                if W[i][j] > 100:
                    W[i][j] = 100

        t1 = time.time()
        total = t1 - t0
        print('the time is:', total)

        minW = np.min(W)
        maxW = np.max(W)
        KK = K*2*np.pi

        f, ax = plt.subplots(figsize=(5,8))
        cc = ax.pcolormesh(THETA,KK,W, cmap ='gnuplot')

        ax.yaxis.set_major_formatter(FuncFormatter(
           lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'
        ))
        ax.yaxis.set_major_locator(MultipleLocator(base=4*np.pi))
        ax.xaxis.set_major_locator(MultipleLocator(base=1))

        cbar = f.colorbar(cc, ax=ax, shrink = .5)
        plt.xlabel('$\Theta$')
        plt.ylabel('K')
    
        this = round(omega,1)
        plt.title('$\Omega =$ %1.2f' % omega)
        cbar.set_label('number of hits')
        cbar.set_ticks([0,100,200])
        cbar.set_ticklabels([0,100,200])

        plt.savefig('bifurcation%d.png' % om)


