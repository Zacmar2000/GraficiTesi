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

    
k=10000

# Estremi intervallo

a=0
b=4

# Consideriamo funzioni in cui il punto medio è l'unico punto di discontinuità in [a,b]

f = lambda x : (np.exp(-x/2)*np.sin(x))*(x<m) + (1/(25*(x-3)**2+1)) *(x>=m)

m=(a+b)/2

d=(b-a)/2

epsilon = np.array([a,m,b]) # Array contenente i punti di discontinuità e gli estremi dell'intervallo in ordine crescente

# Training and test set e relativi campionamenti

X_test = np.linspace(a,b,351)

Y_true = f(X_test)

# Mappa S-Gibbs

S_1 = lambda x: x + k *(x>=m)



N=np.arange(6,80,2)
ERR1=[]
ERR2=[]
ERR3=[]
Lconst1=[]
Lconst2=[]
Lconst3=[]
for n in N:
    X_train = np.linspace(a,b,n+1)
    Y_train = f(X_train)

    # Mappa V_n
    
    V = lambda x: (n*(x-m)/(n-1) + d/(n-1) + m)*(x>=a)*(x<m-(2*d)/n) + (n*(x-m)/( 2*(n-1))+m)*(x<m)*(x>= (m - (2*d)/n)) + x * (x>=m)*(x<=b)
    
    # Fake points
    
    GRASPA_X_train = ScMKTE(V(X_train),epsilon)
    GRASPA_X_test = ScMKTE(V(X_test),epsilon)
    
    Gibbs_X_train = S_1(X_train)
    Gibbs_X_test = S_1(X_test)
    
    
    # Interpolazione
    
    Y_poly = lagrange_interp(X_train,Y_train,X_test)
    Y_GRASPA = lagrange_interp(GRASPA_X_train,Y_train,GRASPA_X_test)
    Y_Gibbs = lagrange_interp(Gibbs_X_train,Y_train,Gibbs_X_test)
    
    # Errori relativi
    
    err1 = np.linalg.norm(Y_true-Y_poly,2)/np.linalg.norm(Y_true,2)
    err2 = np.linalg.norm(Y_true-Y_GRASPA,2)/np.linalg.norm(Y_true,2)
    err3 = np.linalg.norm(Y_true-Y_Gibbs,2)/np.linalg.norm(Y_true,2)
    
    # Funzioni di Lebesgue
    
    lsum1 = lebesgue(X_train,X_test)
    lsum2 = lebesgue(GRASPA_X_train,GRASPA_X_test)
    lsum3 = lebesgue(Gibbs_X_train,Gibbs_X_test)
    
    # Costanti di Lebesgue
    
    lcon1 = np.max(lsum1)
    lcon2 = np.max(lsum2)
    lcon3 = np.max(lsum3)




    ERR1.append(err1)
    ERR2.append(err2)
    ERR3.append(err3)
    
    Lconst1.append(lcon1)
    Lconst2.append(lcon2)
    Lconst3.append(lcon3)


# Grafici

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(17,5))
ax1.cla(); ax2.cla()

ax1.plot(N, Lconst1,'k',linestyle='dashed')
ax1.plot( N, Lconst2,'k',linestyle='solid')
ax1.plot(N , Lconst3,'k',linestyle='dotted')
ax1.set_title("Costante di Lebesgue")
ax1.set_yscale('log')


ax2.plot(N, ERR1,'k',linestyle='dashed')
ax2.plot(N,ERR2,'k',linestyle='solid')
ax2.plot(N,ERR3,'k',linestyle='dotted')
ax2.set_title("Errore relativo")
ax2.set_yscale('log')

ax1.grid(True); ax2.grid(True);
fig.show()
