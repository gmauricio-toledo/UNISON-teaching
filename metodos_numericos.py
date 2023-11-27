import numpy as np


#=============== Regresión ================


#=============== Interpolación ================


#=============== Integración ================


#=============== ODE ================

def runge_kutta_vectorial(f,a,b,h,y0):
    '''
    Función que implementa el método de Runge-Kutta para resolver 
    sistemas de EDO
    '''
    n_eqs = y0.shape[0]
    n = int((b-a)/h)
    xs = np.linspace(a,b,n+1)  # variables independientes
    ys = np.zeros(shape=(n_eqs,xs.shape[0]))
    ys[:,0] = y0
    for j in range(1,n+1):
        k1 = f(xs[j-1],ys[:,j-1])
        k2 = f(xs[j-1]+0.5*h,ys[:,j-1]+0.5*k1*h)
        k3 = f(xs[j-1]+h,ys[:,j-1]-k1*h + 2*k2*h)
        ys[:,j] = ys[:,j-1] + (1/6)*(k1 + 4*k2+k3)*h
    return xs,ys

def print_table(columnas,names:list=None):
    '''
    Función que imprime una tabla con los elementos de los arreglos en renglones
    '''
    pass
    # if names is not None:
    #     assert len(names)==len(columnas)
    #     header = columnas[0]
    #     for name in columnas[1:]:
    #         header += "\t"+name
    #     print(header)
    # for j in range(columnas.shape[1]):
        
    #     print(f"{round(x,3)}\t{round(y,3)}\t{round(y_real,3)}\t{round(er,3)}")