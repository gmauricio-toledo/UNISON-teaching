import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, e
from numpy import sin, cos
from scipy.stats import norm


class EDO_RK_Solver:

    def __init__(self,f,sol=None,order=2,a2=1,n_eqs=1):
        self.f = f
        self.sol = sol
        self.n_eqs = n_eqs
        assert order==2 or order==3 or order==4
        self.order = order
        if order==2:
            self.a2 = a2
            self.a1 = 1-a2
            if a2!=0:
                self.p = 1/(2*a2)
                self.q = 1/(2*a2)
            else:
                self.p = 1
                self.q = 1
        

    def fit(self,a,b,y0,h):
        if y0.shape[0]!=self.n_eqs:
            raise ValueError(f"Number of initial conditions {y0.shape[0]} and number of equations {self.n_eqs} should be equal")
        self.a = a
        self.b = b
        self.h = h
        n = int((b-a)/h)
        xs = np.linspace(a,b,n+1)
        ys = np.zeros(shape=(self.n_eqs,xs.shape[0]))
        ys[:,0] = y0
        if self.order==2:
            for j in range(1,n+1):
                k1 = self.f(xs[j-1],ys[:,j-1])
                k2 = self.f(xs[j-1]+self.p*h,ys[:,j-1]+self.q*k1*h)
                ys[:,j] = ys[:,j-1] + (self.a1*k1 + self.a2*k2)*h 
        if self.order==3:
            pass
        if self.order==4:
            pass
        self.xs = xs
        self.ys = ys

    def plot(self,plot_nodes=True,labels=None):
        xaxis = np.linspace(self.a,self.b,100)
        plt.figure(dpi=100)
        for j in range(self.n_eqs):
            if labels is None:
                label = f'y_{j}'
            else:
                label = labels[j]
            plt.plot(self.xs,self.ys[j,:],label=label)
            if plot_nodes:
                plt.scatter(self.xs,self.ys[j,:],color='black')
        if self.sol is not None:
            plt.plot(xaxis,[self.sol(x) for x in xaxis],color='red',label='Solución real')
            plt.legend(loc='best')
        if self.h>0.2:
            plt.xticks(self.xs)
        plt.legend(loc='best')
        plt.show()   

    def table(self):
        head = "x\t"
        for j in range(self.n_eqs):
            head += f"y_{j}\t"
        print(head)
        for j,x in enumerate(self.xs):
            row = f"{round(x,3)}\t"
            for k in range(self.n_eqs):
                row += f"{round(self.ys[k,j],4)}\t"
            print(row)


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
        if self.h>0.2:
            plt.xticks(self.xs)
        plt.show()      

    def table(self):
        if self.sol is not None:
            valores_reales = self.sol(self.xs)
            errores_relativos = (self.sol(self.xs) - self.ys)/self.sol(self.xs)
            print_table(self.xs,self.ys,valores_reales,errores_relativos)
            self.reales = valores_reales
            self.ers = errores_relativos
        else:
            print("No analytic solution to compare...") 



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


class RungeKutta(EulerEDO):

    def __init__(self,f,sol=None,order=3):
        super().__init__(f,sol)
        assert order==3 or order==4
        self.order = order

    def fit(self,a,b,h,x0,y0):
        assert a==x0        
        self.a = a
        self.b = b
        self.h = h
        n = int((b-a)/h)
        xs = np.linspace(a,b,n+1)
        ys = np.zeros_like(xs)
        ys[0] = y0
        for j in range(1,n+1):
            k1 = self.f(xs[j-1],ys[j-1])
            k2 = self.f(xs[j-1]+0.5*h,ys[j-1]+0.5*k1*h)
            if self.order==3:
                k3 = self.f(xs[j-1]+h,ys[j-1]-k1*h + 2*k2*h)
                ys[j] = ys[j-1] + (1/6)*(k1 + 4*k2+k3)*h
            elif self.order==4:
                k3 = self.f(xs[j-1]+0.5*h,ys[j-1] + 0.5*k2*h)
                k4 = self.f(xs[j-1]+h,ys[j-1] + k3*h)
                ys[j] = ys[j-1] + (1/6)*(k1 + 2*k2+2*k3+k4)*h
        self.xs = xs
        self.ys = ys


def print_table(xs,ys,yrs,ers):
    print("x\ty\ty real\terror rel")
    for x,y,y_real,er in zip(xs,ys,yrs,ers):
        print(f"{round(x,3)}\t{round(y,3)}\t{round(y_real,3)}\t{round(er,3)}")


def load_example(example_number=0):
    examples = {
        0: {
            'a':    0,
            'b':    1,
            'h':    0.25,
            'x0':   0,
            'y0':   1,
            'f':    lambda x,y:(1+4*x)*sqrt(y),
            'sol':  lambda x: 0.25*(x+2*x**2+2)**2,
            'function_f': "(1+4*x)*sqrt(y)"
        },
        1: {
            'a':    0,
            'b':    4,
            'h':    0.25,
            'x0':   0,
            'y0':   1,
            'f':    lambda x,y:-2*x**3 + 12*x**2 - 20*x + 8.5,
            'sol':  lambda x: -0.5*x**4 + 4*x**3 - 10*x**2 + 8.5*x + 1,
            'function_f': "-2*x**3 + 12*x**2 - 20*x + 8.5"
        },
        2: {
            'a':    0,
            'b':    5,
            'h':    0.5,
            'x0':   0,
            'y0':   1,
            'f':    lambda x,y: y*sin(x),
            'sol':  lambda x: (e**(1-cos(x)))
        },
        3: {
            'a':    0,
            'b':    1,
            'h':    0.25,
            'x0':   0,
            'y0':   2,
            'f':    lambda x,y: y*(1-y),
            'sol':  lambda x: 2/(2-e**(-x))
        },
        4: {
            'a':    -1,
            'b':    2.25,
            'h':    0.25,
            'x0':   -1,
            'y0':   e**(5/4),
            'f':    lambda x,y: y*x**3-y,
            'sol':  lambda x: e**(0.25*x**4 - x)
        },
        5: {
            'a':    0,
            'b':    4,
            'h':    0.25,
            'x0':   0,
            'y0':   2,
            'f':    lambda x,y: 4*e**(0.8*x)-0.5*y,
            'sol':  lambda x: e**(-0.5*x)*(3.07692*e**(1.3*x) - 1.07692)
        },
        6: {
            'a':    0,
            'b':    4,
            'h':    0.25,
            'x0':   0,
            'y0':   0.5,
            'f':    lambda x,y: 10*e**(-(1/(2*0.075**2))*(x-2)**2)-0.6*y,
            'sol':  lambda x: e**(-0.6*x)*(3.62402 - 3.12402*norm.cdf(18.888 - 9.42809*x))
        }

    }
    if example_number in examples.keys():
        return examples[example_number]
    else:
        valid_values = list(examples.keys())
        raise KeyError(f'{example_number} is not a valid value: {valid_values}')