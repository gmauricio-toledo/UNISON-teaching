import matplotlib.pyplot as plt
import numpy as np

class EulerEDO:

    def __init__(self,f,sol=None):
        self.f = f
        self.sol = sol

    def fit(self,a,b,h,x0,y0,method='Euler'):
        self.a = a
        self.b = b
        self.h = h
        n = int((b-a)/h)
        xs = np.linspace(a,b,n+1)
        ys = np.zeros_like(xs)
        ys[0] = y0
        if method=='Euler':
            for j in range(1,n+1):
                ys[j] = ys[j-1] + self.f(xs[j-1],ys[j-1])*h
        elif method=='Heun':
            for j in range(1,n+1):
                y_pred = ys[j-1] + self.f(xs[j-1],ys[j-1])*h # Predictor
                ys[j] = ys[j-1] + (self.f(xs[j-1],ys[j-1])+self.f(xs[j],y_pred))*h/2 # Corrector
        elif method=='Mid':
            for j in range(1,n+1):
                y_pred = ys[j-1] + 0.5*self.f(xs[j-1],ys[j-1])*h # Predictor
                x = xs[j-1] + (0.5*h)
                ys[j] = ys[j-1] + self.f(x,y_pred)*h # Corrector
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
        plt.xticks(self.xs)
        plt.show()      

    def table(self):
        if self.sol is not None:
            valores_reales = self.sol(self.xs)
            errores_relativos = (self.sol(self.xs) - self.ys)/self.sol(self.xs)
            print_table(self.xs,self.ys,valores_reales,errores_relativos)
        else:
            print("No hay una solución real") 



class RungeKutta2(EulerEDO):

    def __init__(self,f,sol=None,a2=0.5):
        super().__init__(f,sol) 
        self.a2 = a2
        self.a1 = 1-a2
        if a2!=0:
            self.p = 1/(2*a2)
            self.q = 1/(2*a2)
        else:
            self.p = 1
            self.q = 1

    def fit(self,a,b,h,x0,y0):
        self.a = a
        self.b = b
        self.h = h
        n = int((b-a)/h)
        xs = np.linspace(a,b,n+1)
        ys = np.zeros_like(xs)
        ys[0] = y0
        for j in range(1,n+1):
            k1 = self.f(xs[j-1],ys[j-1])
            k2 = self.f(xs[j-1]+self.p*h,ys[j-1]+self.q*k1*h)
            ys[j] = ys[j-1] + (self.a1*k1 + self.a2*k2)*h 
        self.xs = xs
        self.ys = ys


def print_table(xs,ys,yrs,ers):
    print("x\ty\ty real\terror rel")
    for x,y,y_real,er in zip(xs,ys,yrs,ers):
        print(f"{round(x,3)}\t{round(y,3)}\t{round(y_real,3)}\t{round(er,3)}")