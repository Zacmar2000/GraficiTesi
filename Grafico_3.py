import numpy as np
import matplotlib.pyplot as plt

def C_func(n,Beta,Gamma,x):  #(beta,gamma)-funzione di Chebyshev
    return  np.cos(  ( 2*n*(  np.arccos(x) - Gamma * np.pi /2 ) )/(  2 - Beta - Gamma )   )


def CL_func(n,Beta,Gamma,x):  # funzione avente i (beta,gamma)-punti di CL come zeri
    return  np.sqrt(1-x**2) * np.sin(  ( 2*(n-1)*(  np.arccos(x) - Gamma * np.pi /2 ) )/(  2 - Beta - Gamma )   )


def CL_points(n,Beta,Gamma): #(beta,gamma)-punti di CL
    return np.cos((((2 - Beta - Gamma) * np.arange(0,n) * np.pi)/(2 * (n-1)) ) + (Gamma * np.pi / 2))


def C_points(n,Beta,Gamma): #(beta,gamma)-punti di Chebyshev
    return np.cos((((2 - Beta - Gamma) *(2* np.arange(1,n+1)-1) * np.pi)/(4 * (n)) ) + (Gamma * np.pi / 2))


x = np.arange(-1,1,0.001) # test set

#parametri

n=4
k1=2
k2=3


Beta=2*k1/(n+k1+k2)
Gamma=2*k2/(n+k1+k2)

# (beta,gamma)-punti e funzioni di Chebyshev e CL

y1 = C_func(n,Beta,Gamma,x)
y2 = CL_func(n+1,Beta,Gamma,x)

U = CL_points(n+1,Beta,Gamma)
V = C_points(n,Beta,Gamma)



p1 = C_func(n+k1+k2,0,0,x)
p2 = CL_func(n+k1+k2+1,0,0,x)

U2 = np.concatenate((CL_points(n+k1+k2+1,0,0)[:k2],CL_points(n+k1+k2+1,0,0)[n+k2+1:]))
V2 = np.concatenate((C_points(n+k1+k2,0,0)[:k2],C_points(n+k1+k2,0,0)[n+k2:]))

#grafici

plt.figure()
plt.plot(x,y1,"black")
plt.plot(x,p1,"dimgray",linestyle="dashed")
plt.plot(V2,np.zeros(k1+k2),"rX")
plt.plot(V,np.zeros(n),"bo")
plt.axvline(x = - np.cos(Beta * np.pi /2),color = "red",linestyle = "dotted")
plt.axvline(x =  np.cos(Gamma * np.pi /2),color = "red",linestyle = "dotted")
plt.show()

plt.figure()
plt.plot(x,y2,"black")
plt.plot(x,p2,"dimgray",linestyle="dashed")
plt.plot(U2,np.zeros(k1+k2),"rX")
plt.plot(U,np.zeros(n+1),"bo")
plt.axvline(x = - np.cos(Beta * np.pi /2),color = "red",linestyle = "dotted")
plt.axvline(x =  np.cos(Gamma * np.pi /2),color = "red",linestyle = "dotted")
plt.show()


