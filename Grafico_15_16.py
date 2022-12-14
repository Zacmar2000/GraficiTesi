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

    
n=31
k=10000

# Estremi intervallo

a=0
b=3

# Consideriamo funzioni in cui gli unici punti di discontinuit√† in [a,b] sono in m1 e m2.

f = lambda x : (np.sin(x**2)*np.cos(3*x))*(x<m1) + (1/(25*(x-1.5)**2+(3/2))) *(x>=m1)*(x<m2) + (np.abs(x-(10/4))) *(x>=m2)

m1 = (a+b)/3
m2 = 2*(a+b)/3

epsilon = np.array([a,m1,m2,b]) # Array contenente i punti di discontinuit√† e gli estremi dell'intervallo in ordine crescente

# Training and test set e relativi campionamenti

X_train = np.linspace(a,b,n+1)
X_test = np.linspace(a,b,351)

Y_train = f(X_train)
Y_true = f(X_test)

# Mappa S-Gibbs

S_2 = lambda x: x + k *(x>=m1)*(x<m2) + 2*k*(x>=m2)


# Fake points

GRASPA_X_train = ScMKTE(X_train,epsilon)
GRASPA_X_test = ScMKTE(X_test,epsilon)

Gibbs_X_train = S_2(X_train)
Gibbs_X_test = S_2(X_test)


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




# Grafici

fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(17,5))
ax1.cla(); ax2.cla();ax3.cla()

ax1.plot(X_train, Y_train,'k*', X_test, Y_poly,'b-',X_test,Y_true,'r--')
ax1.set_title("Nodi equispaziati"); 
#ax1.set_xlabel("relative error = %.2e"%err1)

ax2.plot(X_train, Y_train,'k*', X_test, Y_Gibbs,'b-',X_test,Y_true,'r--')
ax2.set_title("mappa S-Gibbs")
#ax2.set_xlabel("relative error = %.2e"%err3)

ax3.plot(X_train, Y_train,'k*', X_test, Y_GRASPA,'b-',X_test,Y_true,'r--')
ax3.set_title("Metodo GRASPA")
#ax3.set_xlabel("relative error = %.2e"%err2)


ax1.set_ylim((-.1, 1.1)); ax2.set_ylim((-.1, 1.1)); ax3.set_ylim((-.1, 1.1));
ax1.set_ylim((-1.1, 1.1)); ax2.set_ylim((-1.1, 1.1));  ax3.set_ylim((-1.1, 1.1));        
ax1.grid(True); ax2.grid(True); ax3.grid(True);
fig.show()



fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(17,5))
ax1.cla(); ax2.cla();ax3.cla()

ax1.plot( X_test, lsum1,'k-')
ax1.set_title("Funzione di Lebesgue per punti equispaziati"); 

ax2.plot( X_test, lsum3,'k-')
ax2.set_title("Funzione di Lebesgue con mappa S-Gibbs"); 

ax3.plot( X_test, lsum2,'k-')
ax3.set_title("Funzione di Lebesgue con metodo GRASPA"); 

ax1.ticklabel_format(style='plain')
ax1.grid(True); ax2.grid(True); ax3.grid(True);
fig.show()
