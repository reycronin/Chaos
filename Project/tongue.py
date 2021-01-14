from mpi4py import MPI
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

t0 = time.time()

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()


def circle(theta, omega, K):
    return (theta + omega - K * np.sin(2 * np.pi * theta)) %1

def get_theta_o(theta_o, omega, K):
    theta_start = []
    theta_start.append(theta_o)
    for i in range(100):
        theta_start.append(circle(theta_start[-1], omega, K))
    return theta_start

def wind(theta_o, omega, K):
    e = .05
    nn = []
    theta_start = get_theta_o(theta_o, omega,K)
    for i in range(len(theta_start)):
        theta = theta_start[i]
        theta_curr = theta + 2*e
        n = 0
        while abs(theta_curr  - theta_start[i]) > e and n < 250:
            theta = circle(theta, omega, K)
            theta_curr = theta
            n += 1
        nn.append(n)
    #print('---------')
    return np.mean(nn)

def remainder(OMEGA, K, theta_o, rem, col, start):
    r = np.zeros((rem, col))
    for i in range(rem):
        for j in range(col):
            k = K[start + rem*rank+i][j]
            omega = OMEGA[start + rem*rank+i][j]
            r[i][j] = wind(theta_o, omega, k)
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
            w[i][j] = wind(theta_o, omega, k)
    return w,r

            
theta_o = .5

dx = .05
y = np.arange(0,1+dx,dx)
x = np.arange(0,1+dx,dx)
OMEGA, K = np.meshgrid(x,y)
W = OMEGA*0.0 
w,r = getwind(OMEGA, K, W, theta_o)

q = comm.gather(w, root = 0)
if rank == 0:
    row = len(K[0])
    num = row//size
    rem = row - size*num
    for i in range(len(q)):
        for j in range(len(q[0])):
           W[(i*num)+j][:] = q[i][j][:]
    if rem != 0 and isinstance(r,int) != True:
        start = row - rem
        for i in range(rem):
            W[start+i][:] = r[i][:]


    t1 = time.time()
    total = t1 - t0
    print('the time is:', total)
    cc = plt.pcolormesh(OMEGA,K,W, cmap ='gnuplot')
    plt.axis('scaled')
    cbar = plt.colorbar()
    plt.xlabel('$\Omega$')
    plt.ylabel('K')
    cbar.set_label('number of iterations where |$\Theta - \Theta_o$| $< \epsilon$')
    plt.savefig('picture.png')

    plt.show()
