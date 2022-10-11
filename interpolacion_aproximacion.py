import numpy as np
import matplotlib.pyplot as plt


class RegresionLineal:
    
    def __init__(self,grado=1):
        self.grado = grado      
        if self.grado > 2:
            print("El grado debe ser 1 o 2.")  

    def fit(self,datos):
        if datos.shape[1]!=2:
            raise ValueError("Matrix should have size n_points x 2")
        self.datos = np.transpose(datos)  # Arreglar esto... que sea de n_points x 2
        self.n = self.datos.shape[1]
        x_prom = np.mean(datos,axis=1)[0]
        y_prom = np.mean(datos,axis=1)[1]
        sum_xy = np.sum(datos[0,:]*datos[1,:])
        sum_x_2s = np.sum(datos[0,:]*datos[0,:])
        sum_x = np.sum(datos[0,:])
        sum_y = np.sum(datos[1,:])
        if self.grado == 1:
            self.a_1 = (self.n*sum_xy - sum_x*sum_y)/(self.n*sum_x_2s - sum_x**2)
            self.a_0 = y_prom - self.a_1*x_prom
            return (self.a_0,self.a_1)
        elif self.grado == 2:
            sum_x_3s = np.sum(datos[0,:]*datos[0,:]*datos[0,:])
            sum_x_4s = np.sum(datos[0,:]*datos[0,:]*datos[0,:]*datos[0,:])
            sum_xxy = np.sum(datos[0,:]*datos[0,:]*datos[1,:])
            A = np.array([[self.n,sum_x,sum_x_2s],[sum_x,sum_x_2s,sum_x_3s],[sum_x_2s,sum_x_3s,sum_x_4s]])
            b = np.array([sum_y,sum_xy,sum_xxy])
            (self.a_0,self.a_1,self.a_2) = tuple(np.dot(np.linalg.inv(A),b))
            return (self.a_0,self.a_1,self.a_2)
    
    def eval(self,x):
        if self.grado == 1:
            return self.a_0 + self.a_1*x
        elif self.grado == 2:
            return self.a_0 + self.a_1*x + self.a_2*x*x

    def plot(self):
        xmin, xmax = np.min(self.datos[0,:]), np.max(self.datos[0,:])
        xaxis = np.linspace(xmin-1,xmax+1,num=50)
        plt.figure(dpi=100)
        plt.scatter(self.datos[0,:],self.datos[1,:],color='black')
        plt.plot(xaxis,[self.eval(x) for x in xaxis],color='red')
        plt.show()

    def metrics(self):
        S_t = np.sum((self.datos[1,:]-np.mean(self.datos,axis=1)[1])**2)
        S_r = np.sum((self.datos[1,:]-self.eval(self.datos[0,:]))**2)
        s_y = np.sqrt(S_t/(self.n-1))
        s_yx = np.sqrt(S_r/(self.n-(self.grado+1)))
        r2 = (S_t-S_r)/S_t
        print(f"Error respecto al promedio: {S_t}")
        print(f"Error cuadrático: {S_r}")
        print(f"S_r<S_t : {S_r<S_t}")
        print(f"Desviación estandar de los datos y_i: {s_y}")
        print(f"Error estandar de la estimación: {s_yx}")
        print(f"S_r<S_t : {s_yx<s_y}")
        print(f"Coeficiente de determinación: {r2}")



class InterpolacionLagrange:
    
    def __init__(self):
        pass

    def fit(self,data,grado=2):
        self.data = data
        if grado <= data.shape[0]-1:
            self.grado = grado
        else:
            print("Grado no valido... se tomará el máximo grado posible")
            self.grado = data.shape[0]-1

    def eval(self,x):
        sum = 0
        for i in range(0,self.grado+1):
            sum += self.__lagrange_p(i,x)*self.data[i,1]
        return sum

    def __lagrange_p(self,i,x):
        prod = 1
        for j in range(self.grado+1):
            if j!=i:
                prod *= (x-self.data[j,0])/(self.data[i,0]-self.data[j,0])
        return prod

    def plot(self):
        x_axis = np.linspace(np.min(self.data,axis=0)[0],np.max(self.data,axis=0)[0],100)
        plt.figure(dpi=200)
        plt.scatter(self.data[:,0],self.data[:,1],color='black',s=50)
        plt.plot(x_axis,[self.eval(x) for x in x_axis])
        plt.show()

class Splines:

    def __init__(self):
        pass

    def fit(self,data,grado=2):
        self.grado = grado
        # ordenar de acuerdo a las coordenadas en x
        self.data = np.array(sorted(data,key=lambda x:x[0]))
        self.num_points = self.data.shape[0]-1
        n = self.num_points
        if self.grado == 1:
            pass
        if self.grado == 2:
            self.A = np.zeros(shape=(3*n,3*n))
            self.b = np.zeros(shape=(3*n,))
            # Llenar A:
            self.A[0,:3] = self.__block(0,order=0) # Extremo izquierdo
            for j in range(1,n):
                pos = j-1
                self.A[2*j-1,3*pos:3*pos+3] = self.__block(j,order=0)
                self.A[2*j,3*(pos+1):3*(pos+1)+3] = self.__block(j,order=0)
            self.A[2*n-1,-3:] = self.__block(n,order=0) # Extremo derecho
            for j in range(1,n):
                pos = 3*(j-1)
                self.A[2*n+j-1,pos:pos+6] = self.__block(j,order=1)
            self.A[3*n-1,0] = 1  # Condición a1=0
            # Llenar b:
            self.b[0] = self.data[0,1]
            for j in range(1,n):
                self.b[2*j-1] = self.data[j,1]
                self.b[2*j] = self.data[j,1] 
            self.b[2*n-1] = self.data[-1,1]
            # Resolver la ecuación:
            W = np.dot(np.linalg.inv(self.A),self.b)
            self.coefs = np.reshape(W,newshape=(-1,3))
            return self.coefs

    def __block(self,i,order):
        if order==0:
            return np.array([self.data[i,0]**2,self.data[i,0],1])
        if order==1:
            return np.array([2*self.data[i,0],1,0,-2*self.data[i,0],-1,0])

    def eval(self,x):
        # determinar en que intervalo está
        idx = np.where(x-self.data[:,0]<0)[0][0]
        if idx == 0 or idx == self.data.shape[0]:
            print("Point to evaluate is outside of interval.")
        else:
            # print(f"Using {idx-1}-th funcion")
            x = np.array([x**2,x,1])
            return np.dot(self.coefs[idx-1,:],x)

    def plot(self):
        x_axis = np.linspace(np.min(self.data,axis=0)[0]+0.01,np.max(self.data,axis=0)[0]-0.01,100)
        plt.figure(dpi=150)
        plt.scatter(self.data[:,0],self.data[:,1],color='black',s=50)
        plt.plot(x_axis,[self.eval(x) for x in x_axis])
        plt.show()

