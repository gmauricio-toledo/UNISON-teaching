import numpy as np
import matplotlib.pyplot as plt


class Regresion_Lineal:
    
    def __init__(self,grado=1):
        self.grado = grado      
        if self.grado > 2:
            print("El grado debe ser 1 o 2.")  

    def fit(self,datos):
        self.datos = datos.copy()
        self.n = datos.shape[1]
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
    
    def line(self,x):
        if self.grado == 1:
            return self.a_0 + self.a_1*x
        elif self.grado == 2:
            return self.a_0 + self.a_1*x + self.a_2*x*x

    def plot(self):
        xmin, xmax = np.min(self.datos[0,:]), np.max(self.datos[0,:])
        xaxis = np.linspace(xmin-1,xmax+1,num=50)
        plt.figure(dpi=100)
        plt.scatter(self.datos[0,:],self.datos[1,:],color='black')
        plt.plot(xaxis,[self.line(x) for x in xaxis],color='red')
        plt.show()

    def metrics(self):
        S_t = np.sum((self.datos[1,:]-np.mean(self.datos,axis=1)[1])**2)
        S_r = np.sum((self.datos[1,:]-self.line(self.datos[0,:]))**2)
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