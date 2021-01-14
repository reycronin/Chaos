import numpy as np
import matplotlib.pyplot as plt

def circle(theta, omega, K):
    return (theta + omega - K * np.sin(2 * np.pi * theta)) 

def wind(theta_o, omega, K):
    n = 0
    e = .03
    theta_1 = theta_o
    nn = []    
    for i in range(1,200):
        for j in range(i):
            theta_1 = circle(theta_1, omega, K)
        theta = theta_1
        while abs(theta - theta_1) < e and n < 250:
            theta = circle(theta, omega, K)
            n += 1
        nn.append(n)
    return np.mean(nn)

def getwind(OMEGA, K, W, theta_o):
    for i in range(len(K)):
        for j in range(len(K[0])):
            k = K[i][j]
            omega = OMEGA[i][j]
            W[i][j] = wind(theta_o, omega, k)
    return W            
            
theta_o = .5

dx = .001
y = np.arange(0,1+dx,dx)
x = np.arange(0,1+dx,dx)
OMEGA, K = np.meshgrid(x,y)
W = OMEGA*0.0 
W = getwind(OMEGA, K, W, theta_o)


plt.figure()
cc = plt.pcolormesh(OMEGA,K,W, cmap ='gnuplot')
plt.axis('scaled')
cbar = plt.colorbar()
plt.xlabel('$\Omega$')
plt.ylabel('K')
cbar.set_label('weighting')
plt.show()
plt.savefig('at.png')



