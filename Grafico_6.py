import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def CL_points(n,Beta,Gamma):
    return np.cos((((2 - Beta - Gamma) * np.arange(0,n) * np.pi)/(2 * (n-1)) ) + (Gamma * np.pi / 2))

def lagrange(x, xx):
    n=len(x); m=len(xx)
    L=np.zeros((m,n))
    for k in range(n):
        x_k = np.concatenate((x[:k],x[k+1:]))
        L[:,k] = np.prod(np.matlib.repmat(xx.reshape(m,1),1,n-1) - np.matlib.repmat(x_k,m,1) , axis=1) / np.prod(x[k]-x_k)
    return L


def lagrange_interp(x,y,xx):
    L = lagrange(x,xx)
    return np.dot(L,y)

# Funzione

f= lambda x : (1/(1+25*x**2))

# Test set

XX=np.arange(-1,1,0.0005)
YY=f(XX)



# Calcolo errori assoluti

N=np.arange(5,52,2)

Err_ass = []
Err_ass_eq = []

for n in N:
    beta= 1/((n+1))
    gamma= 1/((n+1))

    x1 = np.linspace(-1,1,n)

    x2 = CL_points(n,beta,gamma)


    y1 = f(x1)
    y2 = f(x2)

    y_eq = lagrange_interp(x1,y1,XX)
    y_CL = lagrange_interp(x2,y2,XX)

    err_ass_eq = np.linalg.norm(YY-y_eq,np.inf)
    err_ass = np.linalg.norm(YY-y_CL,np.inf)

    Err_ass.append(err_ass)
    Err_ass_eq.append(err_ass_eq)

# Interpolazione per n=40

n=40
beta= 1/((n+1))
gamma= 1/((n+1))

x1 = np.linspace(-1,1,n)

x2 = CL_points(n,beta,gamma)


y1 = f(x1)
y2 = f(x2)

p1 = lagrange_interp(x1,y1,XX)
p2 = lagrange_interp(x2,y2,XX)

# Grafici

plt.figure()
plt.plot(XX,p2,"black")
plt.show()


plt.figure()
plt.yscale("log")
plt.plot(N,Err_ass,"black")
plt.plot(N,Err_ass_eq,"red",linestyle = "dashed")
plt.show()
