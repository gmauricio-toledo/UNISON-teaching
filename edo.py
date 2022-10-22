import matplotlib.pyplot as plt
import numpy as np

class EulerEDO:

    def __init__(self,f,sol=None):
        self.f = f
        self.sol = sol

    def fit(self,a,b,h,x0,y0,heun=False):
        self.a = a
        self.b = b
        self.h = h
        n = int((b-a)/h)
        xs = np.linspace(a,b,n+1)
        ys = np.zeros_like(xs)
        ys[0] = y0
        if not heun:
            for j in range(1,n+1):
                ys[j] = ys[j-1] + self.f(xs[j-1])*h
        else:
            for j in range(1,n+1):
                ys[j] = ys[j-1] + (self.f(xs[j-1])+self.f(xs[j]))*h/2
        self.xs = xs
        self.ys = ys
    
    def plot(self,plot_nodes=True):
        xaxis = np.linspace(self.a,self.b,100)
        plt.figure(dpi=100)
        plt.plot(self.xs,self.ys,color='blue',label='Solución aprox')
        if plot_nodes:
            plt.scatter(self.xs,self.ys,color='blue')
        if self.sol is not None:
            plt.plot(xaxis,[self.sol(x) for x in xaxis],color='red',label='Solución real')
            plt.legend(loc='best')
        plt.show()    


class EulerEDOxy:

    def __init__(self,f,sol=None):
        self.f = f
        self.sol = sol

    def fit(self,a,b,h,x0,y0,heun=False):
        self.a = a
        self.b = b
        self.h = h
        n = int((b-a)/h)
        xs = np.linspace(a,b,n+1)
        ys = np.zeros_like(xs)
        ys[0] = y0
        if not heun:
            for j in range(1,n+1):
                ys[j] = ys[j-1] + self.f(xs[j-1],ys[j-1])*h
        else:
            for j in range(1,n+1):
                y_pred = ys[j-1] + self.f(xs[j-1],ys[j-1])*h
                ys[j] = ys[j-1] + (self.f(xs[j-1],ys[j-1])+self.f(xs[j],y_pred))*h/2
        self.xs = xs
        self.ys = ys
    
    def plot(self,plot_nodes=True):
        xaxis = np.linspace(self.a,self.b,100)
        plt.figure(dpi=100)
        plt.plot(self.xs,self.ys,color='blue',label='Solución aprox')
        if plot_nodes:
            plt.scatter(self.xs,self.ys,color='blue')
        if self.sol is not None:
            plt.plot(xaxis,[self.sol(x) for x in xaxis],color='red',label='Solución real')
            plt.legend(loc='best')
        plt.show()       


def print_table(xs,ys,yrs,ers):
    print("x\ty\ty real\terror rel")
    for x,y,y_real,er in zip(xs,ys,yrs,ers):
        print(f"{round(x,3)}\t{round(y,3)}\t{round(y_real,3)}\t{round(er,3)}")