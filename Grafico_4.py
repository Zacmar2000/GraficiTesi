import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


def CL_points(n,Beta,Gamma): #(beta,gamma)-punti di CL
    return np.cos((((2 - Beta - Gamma) * np.arange(0,n) * np.pi)/(2 * (n-1)) ) + (Gamma * np.pi / 2))



def lagrange(U, x): 
    n = len(U)
    m = len(x)
    Lagr = np.zeros((m,n))
    for k in range(n):
        U_temp = np.concatenate((U[:k],U[k+1:]))
        Lagr[:,k] = np.prod(np.matlib.repmat(x.reshape(m,1),1,n-1) - np.matlib.repmat(U_temp,m,1) , axis=1) / np.prod(U[k]-U_temp)
    return Lagr

def C_Leb(U, x): # Costante di Lebesgue
    Leb = np.abs( lagrange(U,x) )
    return np.max(np.sum( Leb, axis = -1 ))



X = np.arange(-1,1,0.001) # Test set


n = 30



# Caso con beta in [0,1/n)

beta1 = np.arange(0,1/n,0.0007)
y = []


for b in beta1:
    U = CL_points(n,b,0)
    y.append(C_Leb(U,X))


plt.figure()
plt.plot(beta1,y,"black")
plt.show()



# Caso con beta in [1/n,3/n)

beta2 = np.arange(1/n,3/n,0.001)
y2=[]


for b in beta2:
    U = CL_points(n,b,0)
    y2.append(C_Leb(U,X))

plt.figure()
plt.plot(beta2,y2,"black")
plt.show()
