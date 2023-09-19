import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import fabs
import matplotlib.animation as animation
import itertools

class RootFinder:

    def __init__(self,f):
        self.root = None
        self.f = f
        self.methods = ['bisection',
                        'false-position',
                        'fix',
                        'newton',
                        'secant'
                        ]
        print(f"Possible methods: {self.methods}")
        self.plot(bracket=[-3,3])

    def fit(self,tolerance,x0,x1=None,method='bisection',df=None):
        self.x0 = x0
        self.x1 = x1
        self.root = x0
        self.df = df
        self.tolerance = tolerance
        self.method = method
        self.n_iterations = 0
        self.errors = []
        self.roots = []
        self.__check()
        error = 2*self.tolerance
        while (error>self.tolerance) and self.n_iterations<10000:
            root_old = self.root
            if method=='bisection':
                self.root = 0.5*(x0 + x1)
                test = self.f(x0)*self.f(self.root)
                if test<0:        
                    x1 = self.root
                elif test>0:
                    x0 = self.root
                else:
                    error = 0
            elif method=='false-position':
                self.root = x1 - (self.f(x1)*(x0-x1))/(self.f(x0)-self.f(x1))
                test = self.f(x0)*self.f(self.root)
                if test<0:        
                    x1 = self.root
                elif test>0:
                    x0 = self.root
                else:
                    error = 0
            elif method=='newton':
                self.root = root_old - self.f(root_old)/self.df(root_old)
            elif method=='secant':
                self.root = x1 - self.f(x1)*(x0-x1)/(self.f(x0)-self.f(x1))
                x0 = x1
                x1 = self.root
            elif method=='fix':
                root_old = self.root
                self.root = self.f(self.root)+self.root
            self.roots.append(self.root)
            if self.root != 0:
                error = self.__relative_error(self.root,root_old)
            self.errors.append(error)
            self.n_iterations += 1
        if self.n_iterations >= 10000:
            print("Iteraciones máximas alcanzadas")
        return self.root
    
    def __check(self):
        assert self.method in self.methods, f"Method '{self.method}' not valid"
        if self.method in ['false-position','secant']:
            assert self.x1>self.x0, f"x1={self.x1} should be greater than x0={self.x0}"
        if self.method == 'newton':
            assert self.df is not None, "Derivative of f is not given"
    
    def __relative_error(self,x_real,x_aprox):
        return abs(x_real-x_aprox)/abs(x_real) 

    def plot(self,bracket=None,window=5):
        if self.root is not None:
            x_axis = np.linspace(self.root-0.5*window,self.root+0.5*window,100)
            plt.figure(dpi=100)
            plt.plot(x_axis,[self.f(x) for x in x_axis],color='black')
            plt.axhline(0)
            plt.scatter([self.root],[self.f(self.root)])
            plt.show()
        elif isinstance(bracket,list):
            x_axis = np.linspace(bracket[0],bracket[1],100)
            plt.figure(dpi=100)
            plt.plot(x_axis,[self.f(x) for x in x_axis],color='black')
            plt.axhline(0)
            plt.show()
        else:
            print("Run 'fit' method first or specify a valid bracket")

    def animate(self,fname):
        def update(i):
            ax.clear()
            fig.suptitle(f"Iteración {i}")
            ax.axhline(0,color='gray')
            ax.plot(eje_x,eje_y,color='red')
            ax.scatter([next(data_x)], [next(data_y)],s=50,color='black')
        data_x = itertools.cycle(self.roots)
        data_y = itertools.cycle(np.zeros_like(self.roots))
        frames = len(self.roots)
        if self.method in ['false-position','bisection']: 
            window_width = fabs(self.x1-self.x0)
            eje_x = np.linspace(start=self.x0-0.1*window_width,
                                stop=self.x1+0.1*window_width,
                                num=100)
            eje_y = [self.f(x) for x in eje_x] 
            fig, ax = plt.subplots(figsize=(7,5))
            ani = animation.FuncAnimation(fig, update, frames=frames, interval=500)
            ani.save(fname, writer='pillow')
        elif self.method in ['fix','newton','secant']:
            a, b = np.min(self.roots), np.max(self.roots)
            window_width = fabs(b-a)
            eje_x = np.linspace(start=a-0.1*window_width,
                                stop=b+0.1*window_width,
                                num=100)
            eje_y = [self.f(x) for x in eje_x] 
            fig, ax = plt.subplots(figsize=(7,5))
            ani = animation.FuncAnimation(fig, update, frames=frames, interval=500)
            ani.save(fname, writer='pillow')
        else:
            print("Something's wrong")

    def print_table(self,digits=5):
        data = {'n_iter': [j+1 for j,x in enumerate(self.roots)],
                'root': [round(x,digits) for x in self.roots],
                'error': [round(self.errors[j],digits) for j,x in enumerate(self.roots)],
                'f(x)': [round(self.f(x),digits) for x in self.roots]
                }
        display(pd.DataFrame(data=data))