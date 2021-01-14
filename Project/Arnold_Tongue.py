#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt

def circle(theta, omega, K):
    return theta + omega - K * np.sin(2 * np.pi * theta)

def wind(theta_o, omega, K):
    theta = theta_o
    n = 0
    print('----------------')
    while n < 300:
        theta = circle(theta, omega, K)
        print('theta: ', theta)
        n += 1
    return ((theta - theta_o)/n) % 1

def getwind(OMEGA, K, W, theta_o):
    for i in range(len(K)):
        for j in range(len(K[0])):
            k = K[i][j]
            omega = OMEGA[i][j]
            W[i][j] = wind(theta_o, omega, k)
    return W            
            
theta_o = 0.5

dx = .005
y = np.arange(0,1+dx,dx)
x = np.arange(0,1+dx,dx)
OMEGA, K = np.meshgrid(x,y)
W = OMEGA*0.0 
W = getwind(OMEGA, K, W, theta_o)

cc = plt.pcolormesh(OMEGA,K,W, cmap ='gnuplot')
plt.axis('scaled')

cbar = plt.colorbar()
plt.xlabel('$\Omega$')
plt.ylabel('K')
cbar.set_label('weighting')


# In[329]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt

def circle(theta, omega, K):
    return (theta + omega - K * np.sin(2 * np.pi * theta)) 

def wind(theta_o, omega, K):
    theta = theta_o
    n = 0
    temp = circle(theta,omega,K)
    e = .003
    while abs(theta - theta_o) < e and n < 250:
        theta = circle(theta, omega, K)
        n += 1
    return n

def getwind(OMEGA, K, W, theta_o):
    for i in range(len(K)):
        for j in range(len(K[0])):
            k = K[i][j]
            omega = OMEGA[i][j]
            W[i][j] = wind(theta_o, omega, k)
    return W            
            
theta_o = .5

dx = .1
y = np.arange(0,2+dx,dx)
x = np.arange(0,1+dx,dx)
OMEGA, K = np.meshgrid(x,y)
W = OMEGA*0.0 
W = getwind(OMEGA, K, W, theta_o)



cc = plt.pcolormesh(OMEGA,K,W, cmap ='gnuplot')
plt.axis('scaled')

cbar = plt.colorbar()
plt.xlabel('$\Omega$')
plt.ylabel('K')
cbar.set_label('weighting')


# In[327]:


print(np.max(W))


# In[288]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt

def circle(theta, omega, K):
    return theta + omega - K * np.sin(2 * np.pi * theta) / (2 * np.pi)

def wind(theta_o, omega, K):
    theta = theta_o
    n = 0
    while n < 250:
        theta = circle(theta, omega, K)
        n += 1
    return ((theta - theta_o)/n) % 1

def getwind(OMEGA, K, W, theta_o):
    for i in range(len(OMEGA)):
        omega = OMEGA[i]
        W[i] = wind(theta_o, omega, K)
    return W            
            
theta_o = .5
K = 1
omega = np.linspace(0,1,200)
W = omega * 0.0 
W = getwind(omega, K, W, theta_o)

plt.plot(omega, W, 'k')
plt.title('devils staircase')
plt.xlabel('omega')
plt.ylabel('W')


# In[274]:


x = np.linspace(0,1,200)
print(x)


# In[164]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#plt.style.use('ggplot')

omega = []
sigma = []



def logistic(theta_o):
    omega.append(0)
    om = .01
    i = .01
    while om <= 1:
        om = omega[-1] + i
        K = 1-10E-3
        theta = theta_o
        n = 0
        diff = 0
        while diff < .4 and n < 500:
            temp = theta
            theta = (theta + omega[-1] - K * np.sin(2 * np.pi * theta) / (2 * np.pi)) % 1
            diff = theta - theta_o
            n += 1
        omega.append(om)
        s = theta/n
        sigma.append(s)
    del(omega[-1])
    return(K, omega, sigma)

K, omega, sigma = logistic(.3)




plt.plot(omega, sigma, 'k')
plt.xlabel('omega')
plt.ylabel('W')


# In[106]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#plt.style.use('ggplot')

omega = []
sigma = []


def logistic(theta_o):
    omega.append(0)
    K = 1 - 10E-3   
    om = 0
    i = .001
    while om <= 1:
        om = omega[-1] + i
        theta = theta_o
        n = 0
        while theta <= 1 and theta >= 0 and n < 100:
            temp = theta
            theta = theta + omega[-1] - K * np.sin(2 * np.pi * theta)/(2*np.pi)
            n += 1
        print(n)
        print(temp)
        sigma.append((theta-temp)/(n-1))
        omega.append(om)
    del(omega[-1])
    return(K, omega, sigma)

K, omega, sigma = logistic(.3)


plt.plot(omega, sigma, 'k.')


# In[161]:


theta = .5
theta = (theta + .1 - 1 * np.sin(2 * np.pi * theta) / (2 * np.pi)) % 1
theta = .2%1
print(theta)


# In[291]:



# plot widing numbers for a circle map 
# 
import numpy as np
import matplotlib.pyplot as plt

# function without moding
def cf(theta,Omega,K):
	twopi = 2.0*np.pi
	x = theta + Omega - (K/twopi)*np.sin(twopi*theta)  
	return x

# circle map is theta_{n+1} = theta_n + Omega  - K/(2pi)*sin(2 pi theta_n)
# mod 1
def circlemap(theta,Omega,K):
	x = cf(theta,Omega,K) 
	return np.fmod(x,1.0)   # returned in [0,1)


# compute the winding number 
# the winding number is lim n to infty sum theta_n/n
# don't use mod
# initial value of theta is theta0 
# Omega, K are passed to circle map function cf
def winding(theta0,Omega,K,nit):
	theta = theta0
	for i in range(0,nit):  # range goes to n-1
		theta = cf(theta,Omega,K)    # no mods

	sum = (theta-theta0)/float(nit)
	return np.mod(sum,1.0)

# fill up a 2x2 array with winding numbers
# here X is Omega array
# and Y is K array
def fillwinding(X,Y):
	nr= len(X)
	nc= len(X[0])
	WW = X*0.0 + 0.0
	theta0 = 0.1
	for i in range(0,nr):
		for j in range(0,nc):
			xij = X[i][j]   # Omega 
			yij = Y[i][j]   # K
			WW[i][j] = winding(theta0,xij,yij,200)


	return WW


# make a mesh, note x,y are now 2d arrays containing all x values and all y values
ngrid = 200.0
xmax = 1.0   # x is omega
dx = xmax/ngrid
ymax = 2.0*np.pi  # y is K
dy = ymax/ngrid
X,Y = np.meshgrid(np.arange(0.0,xmax+dx,dx),np.arange(0.0,ymax+dy,dy))
WW = fillwinding(X,Y)

plt.figure()
plt.xlabel('Omega')
plt.ylabel('K')
plt.axis([0.0, xmax, 0.0, ymax])

# plot with color as an image:
cc = plt.pcolormesh(X,Y,WW)
plt.colorbar()


# In[248]:


print(len(X[0]))


# In[198]:


ngrid = 200.0
xmax = 1.0   # x is omega
dx = xmax/ngrid
ymax = 2.0*np.pi  # y is K
dy = ymax/ngrid
X,Y = np.meshgrid(np.arange(0.0,xmax+dx,dx),np.arange(0.0,ymax+dy,dy))
print(X[1][2])
print(dy)


# In[201]:


k = np.arange(0.0,np.pi+.001,.001)
omega = np.arange(0.0,1.0+.001,.001)
OMEGA, K = np.meshgrid(omega,k)
print(len(OMEGA))

