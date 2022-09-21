import numpy as np

class RootFinder:

    def __init__(self):
        pass

    def fit(self,f,x0,x1,tolerance,method='bisection'):
        self.x0 = x0
        self.x1 = x1
        self.root = x0
        self.f = f
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
            elif method=='false-position':
                self.root = x1 - (self.f(x1)*(x0-x1))/(self.f(x0)-self.f(x1))
            self.roots.append(self.root)
            if self.root != 0:
                error = self.__relative_error(self.root,root_old)
            test = self.f(x0)*self.f(self.root)
            if test<0:        
                x1 = self.root
            elif test>0:
                x0 = self.root
            else:
                error = 0
            self.errors.append(error)
            self.n_iterations += 1
        return self.root
    
    def __check(self):
        assert self.x1>self.x0
        assert self.method in ['bisection',
                                'false-position',
                                'fix',
                                'newton',
                                'secant']
    
    def __relative_error(self,x_real,x_aprox):
        return abs(x_real-x_aprox)/abs(x_real) 