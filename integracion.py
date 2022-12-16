import matplotlib.pyplot as plt
import numpy as np
from interpolacion_aproximacion import InterpolacionLagrange

class Trapezoid:

    def __init__(self,f=None,datos=None):
        if f is not None:
            self.f = f
            self.function = True
            self.table = False
        if datos is not None:
            self.datos = datos.copy()
            self.function = False
            self.table = True

    def fit(self,a,b,n):
        '''
        Tal vez revisar los de kwargs para sólo definir como fit(self, kwargs) o algo así
        '''
        if self.function:
            self.n = n
            if a!=b:
                if a<b:
                    self.a = a
                    self.b = b
                elif b<a:
                    print(f"It should be {a}<{b}... considering the interval as [{b},{a}].")
                    self.a,self.b = b,a
                h = (self.b-self.a)/self.n
                sum = 0
                for j in range(1,self.n):
                    sum += self.f(self.a+j*h)
                self.aprox = (self.b-self.a)*(self.f(self.a)+2*sum+self.f(self.b))/(2*self.n)    
            else:
                self.aprox = 0
        elif self.table:
            print("a,b,n will be ignored.")
            n_points = self.datos.shape[0]
		a = self.datos[0,0]
		b = self.datos[-1,0]
            if a!=b:
                if a<b:
                    self.a = a
                    self.b = b
                elif b<a:
                    print(f"It should be {a}<{b}... considering the interval as [{b},{a}].")
                    self.a,self.b = b,a
                self.aprox = 0
                for j in range(1,n_points):
                    self.aprox += (self.datos[j,0]-self.datos[j-1,0])*(self.datos[j-1,1]+self.datos[j,1])/2    
            else:
                self.aprox = 0
        return self.aprox

    def plot(self):
        x_axis = np.linspace(self.a,self.b,100)
        division = np.linspace(self.a,self.b,self.n+1)
        plt.figure(dpi=100)
        plt.plot(x_axis,[self.f(x) for x in x_axis])
        plt.plot(x_axis,[0 for x in x_axis],color='gray')
        plt.plot(division,[self.f(x) for x in division])
        for x in division:
            plt.plot([x,x],[0,self.f(x)],color='gray')
        plt.show()

    def get_errors(self,dx2_prom,real_value=None,silent=False):
        self.aprox_error = -dx2_prom*(self.b-self.a)**3/(12*self.n**2)
        if not silent:
            print(f"Error aproximado: {self.aprox_error}")
        if real_value is not None:
            self.real_error = (real_value - self.aprox)/real_value
            if not silent:
                print(f"Error relativo: {self.real_error}")

class Simpson:

    def __init__(self,f):
        self.f = f
        pass

    def fit(self,a,b,n):
        self.n = n
        if n==1:
            if a!=b:
                if a<b:
                    self.a = a
                    self.b = b
                elif b<a:
                    print(f"It should be {a}<{b}... considering the interval as [{b},{a}].")
                    self.a,self.b = b,a
                xm = (self.b+self.a)/2
                self.nodes = np.array([self.a,xm,self.b])
                self.aprox = (self.b-self.a)*(self.f(self.a)+4*self.f(xm)+self.f(self.b))/6    
            else:
                self.aprox = 0
        else:
            assert n%2 == 0
            if a!=b:
                if a<b:
                    self.a = a
                    self.b = b
                elif b<a:
                    print(f"It should be {a}<{b}... considering the interval as [{b},{a}].")
                    self.a,self.b = b,a
                h = (self.b-self.a)/self.n
                self.nodes = np.linspace(self.a,self.b,self.n+1)
                sum_even = 0
                sum_odd = 0
                for j in range(n//2):
                    sum_odd += self.f(a + (2*j+1)*h)
                for j in range(n//2):
                    sum_even += self.f(a + (2*j)*h)
                self.aprox = (self.b-self.a)*(self.f(self.a)+4*sum_odd+2*sum_even+self.f(self.b))/(3*self.n)    
            else:
                self.aprox = 0
        return self.aprox

    def plot(self):
        x_axis = np.linspace(self.a,self.b,100)
        data = np.zeros(shape=(self.nodes.shape[0]//2,3))
        for k in range(self.nodes.shape[0]//2):
            data[k] = self.nodes[2*k:2*k+3]
        plt.figure(dpi=100)
        plt.plot(x_axis,[self.f(x) for x in x_axis])
        plt.plot(x_axis,[0 for x in x_axis],color='gray')
        
        for triplet in data:
            values = np.array([self.f(x) for x in triplet])
            triplet_data = np.transpose(np.vstack((triplet,values)))
            lag = InterpolacionLagrange()
            lag.fit(triplet_data,grado=2)
            sub_interval = np.linspace(triplet[0],triplet[2],50)
            plt.plot(sub_interval,[lag.eval(x) for x in sub_interval],color='red')
        for x in self.nodes:
            plt.plot([x,x],[0,self.f(x)],color='gray')
        plt.show()

    def get_errors(self,dx4_prom,real_value=None,silent=False):
        self.aprox_error = -dx4_prom*(self.b-self.a)**5/(2880*self.n**4)
        if not silent:
            print(f"Error aproximado: {self.aprox_error}")
        if real_value is not None:
            self.real_error = (real_value - self.aprox)/real_value
            if not silent:
                print(f"Error relativo: {self.real_error}")