import numpy as np
import matplotlib.pyplot as plt

class RootFinder:

    def __init__(self):
        self.root = None

    def fit(self,f,tolerance,x0,x1=None,method='bisection',df=None):
        self.x0 = x0
        self.x1 = x1
        self.root = x0
        self.f = f
        self.df = df
        self.tolerance = tolerance
        self.method = method
        self.n_iterations = 0
        self.errors = []
        self.roots = []
        self.__check()
        error = 2*self.tolerance
        while (error>self.tolerance):
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
                # reasignar
            elif method=='newton':
                self.root = root_old - self.f(root_old)/self.df(root_old)
            elif method=='secant':
                self.root = x1 - f(x1)*(x0-x1)/(f(x0)-f(x1))
                x0 = x1
                x1 = self.root
            elif method=='fix':
                pass
            self.roots.append(self.root)
            if self.root != 0:
                error = self.__relative_error(self.root,root_old)
            self.errors.append(error)
            self.n_iterations += 1
        return self.root
    
    def __check(self):
        assert self.method in ['bisection',
                                'false-position',
                                'fix',
                                'newton',
                                'secant']
        if self.method in ['false-position','secant']:
            assert self.x1>self.x0
    
    def __relative_error(self,x_real,x_aprox):
        return abs(x_real-x_aprox)/abs(x_real) 

    def plot(self,window=5):
        if self.root is not None:
            x_axis = np.linspace(self.root-window,self.root+window,100)
            plt.figure(dpi=100)
            plt.plot(x_axis,[self.f(x) for x in x_axis],color='black')
            plt.plot(x_axis,[0 for x in x_axis],color='gray')
            plt.scatter([self.root],[self.f(self.root)])
            plt.show()
        else:
            print("Run 'fit' method first")

    def print_table(self):
        pass
        # head = "x\t"
        # head += f"y_{j}\t"
        # print(head)
        # for j,x in enumerate(self.xs):
        #     row = f"{round(x,3)}\t"
        #     for k in range(self.n_eqs):
        #         row += f"{round(self.ys[k,j],4)}\t"
        #     print(row)