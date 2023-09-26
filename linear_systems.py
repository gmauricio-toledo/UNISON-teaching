import numpy as np
from math import floor

def find_example(dim,norma_coefs=10,norma_sol=20):
    norma_coefs = floor(norma_coefs)
    opciones = list(range(-norma_coefs,norma_coefs+1))
    found = False
    while not found:
        A = np.random.choice(opciones,size=(dim,dim),replace=True)
        b = np.random.choice(opciones,size=(dim,),replace=True)
        try:
            X = np.linalg.inv(A)@b
        except:
            pass
        if are_integers(X) and is_smaller_enough(X,norma_sol):
            print("Matriz A:\n",A)
            print("Lado derecho b:\n",b.reshape((dim,1)))
            print("Solución:\n",X)
            found = True
    return A,b,X

def are_integers(X):
    '''
    Función para verificar que todas las entradas de X sean enteras
    '''
    s = False
    for x in X:
        if x-floor(x)==0:
            s = True
        else:
            s = False
            break
    return s

def is_smaller_enough(X,norma):
    '''
    Función para verificar que todas las entradas de X sean menores que una norma
    '''
    return bool(np.prod(np.abs(X)<norma))


def eliminacion_gaussiana(A,b,resultados_parciales=False):
    n = A.shape[0]
    assert b.shape[0] == n
    A = np.hstack((A,b.reshape(-1,1)))
    if resultados_parciales: print("Inicial:\n", A)
    for k in range(n-1):
        for i in range(k+1,n):
            ratio = A[i,k]/A[k,k]
            A[i] = A[i]-ratio*A[k]
            if resultados_parciales: print(A)
    if resultados_parciales: print("Forward elimination:\n",A)
    variables = A[:,n]
    for k in reversed(range(n)):
        for j in range(k+1,n):
            variables[k] = variables[k] - A[k,j]*variables[j] 
        variables[k] = variables[k]/A[k,k]
    print("Valor de las variables:\n",variables)
    return variables

