import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings("ignore")

def CL_points(n,Beta,Gamma):
    return np.cos((((2 - Beta - Gamma) * np.arange(0,n) * np.pi)/(2 * (n-1)) ) + (Gamma * np.pi / 2))

def lagrange(x, xx):
    n=len(x); m=len(xx)
    L=np.zeros((m,n))
    for k in range(n):
        x_k = np.concatenate((x[:k],x[k+1:]))
        L[:,k] = np.prod(np.matlib.repmat(xx.reshape(m,1),1,n-1) - np.matlib.repmat(x_k,m,1) , axis=1) / np.prod(x[k]-x_k)
    return L

def lebesgue(x, xx):
    L = np.abs( lagrange(x,xx) )
    return np.sum( L, axis = -1 )

def lagrange_interp(x,y,xx):
    L = lagrange(x,xx)
    return np.dot(L,y)

def MKTE(x,epsilon): 
    F = lambda X,e0,e1 : 2*(X-e0)/(e1-e0) -1
    G = lambda X,e0,e1 : (e1 - e0)*(X+1)/2 + e0
    M1 = lambda X: np.sin( np.pi * X /2)
    for i in range(1,len(epsilon)):
         if x>=epsilon[i-1] and x< epsilon[i] :
             M = G(M1(F(x,epsilon[i-1],epsilon[i])),epsilon[i-1],epsilon[i])
    if x == epsilon[-1]:
        M = epsilon[-1]
    return M

def ScompMKTE(x,epsilon):
    S = lambda X,i : X + (i-1)*k
    for i in range(1,len(epsilon)):
         if x>=epsilon[i-1] and x< epsilon[i] :
             sol_scommkte = S(MKTE(x,epsilon),i)
    if x == epsilon[-1]:
        sol_scommkte = S(epsilon[-1],len(epsilon)-1)
    return sol_scommkte

def ScMKTE(X,epsilon): # Composizione della mappa S-Gibbs e della mappa MKTE
    y=[]
    for x in X:
        y.append(ScompMKTE(x,epsilon))
    return np.array(y)

def p_2_1(x,xx):
  xx_2 = xx[(xx>=0)&(xx<=1)]
  i1=int((len(x)+1)/3)
  x_1 = x[:i1]
  x_2 = x[i1+1:-i1]
  x_i = np.concatenate((x_1,x_2))
  n=len(x_i); m=len(xx_2)
  R=np.zeros(m)
  R[:]= np.prod(np.matlib.repmat(xx_2.reshape(m,1),1,n) - np.matlib.repmat(x_i,m,1) , axis=1) / np.prod(x[i1]-x_i)
  return np.max(np.abs(R))

def C_2_1(n):
  return 2**((n+2)/3)

    
k=10000

# Estremi intervallo

a=0
b=3

# Consideriamo funzioni in cui gli unici punti di discontinuità in [a,b] sono in m1 e m2.

f = lambda x : (np.sin(x**2)*np.cos(3*x))*(x<m1) + (1/(25*(x-1.5)**2+(3/2))) *(x>=m1)*(x<m2) + (np.abs(x-(10/4))) *(x>=m2)

m1 = (a+b)/3
m2 = 2*(a+b)/3

epsilon = np.array([a,m1,m2,b]) # Array contenente i punti di discontinuità e gli estremi dell'intervallo in ordine crescente

# test set e relativi campionamenti

X_test = np.linspace(a,b,351)

Y_true = f(X_test)

# Mappa S-Gibbs

S_2 = lambda x: x + k *(x>=m1)*(x<m2) + 2*k*(x>=m2)



N=np.arange(4,110,3)
Lconst2=[]
P_C_2_1=[]
for n in N:

    # Training set
    X_train = np.linspace(a,b,n+1)
    Y_train = f(X_train)
    
    
    # Fake points
    
    GRASPA_X_train = ScMKTE(X_train,epsilon)
    GRASPA_X_test = ScMKTE(X_test,epsilon)

    # Funzione di Lebesgue

    lsum2 = lebesgue(GRASPA_X_train,GRASPA_X_test)
    
    # Costante di Lebesgue
    
    lcon2 = np.max(lsum2)

    # Calcolo di p*C 

    p_21 = p_2_1(GRASPA_X_train,X_test)
    p_c_21 = p_21*C_2_1(n)



    P_C_2_1.append(p_c_21)
    
    Lconst2.append(lcon2)



# Grafici

fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(17,5))
ax1.cla(); ax2.cla()

ax1.plot( N, Lconst2,'k',linestyle='solid')
ax1.set_title("Costante di Lebesgue")
ax1.set_yscale('log')

ax2.plot( N, P_C_2_1,'k',linestyle='solid')
ax2.set_title('"Fattore aggiuntivo"')
ax2.set_yscale('log')


ax1.grid(True); ax2.grid(True);
fig.show()

