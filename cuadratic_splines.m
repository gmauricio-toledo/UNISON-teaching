clc;
clear -global;
global data = [3,2.5;4.5,1;7,2.5;9,0.5] # Definimos los datos

function x = bloque_3x1(j)
  global data;
  x = [data(j,1)^2,data(j,1),1];
endfunction

function x = bloque_6x1(j)
  global data;
  x = [2*data(j,1),1,0,-2*data(j,1),-1,0];
endfunction

num_points = rows(data);
n = num_points-1;
A = zeros(3*n,3*n);
b = zeros(3*n,1);

# ------ Fill A -------
A(1,1:3) = bloque_3x1(1); # Extremo izquierdo
for j = 1:n-1
  pos = j-1;
  A(2*j,3*pos+1:3*pos+3) = bloque_3x1(j+1);
  A(2*j+1,3*(pos+1)+1:3*(pos+1)+3) = bloque_3x1(j+1);
 endfor
A(2*n,3*n-2:end) = bloque_3x1(num_points); # Extremo derecho
for j = 1:n-1     # Condiciones derivada 
    pos = 3*(j-1);
    A(2*n+j,pos+1:pos+6) = bloque_6x1(j+1);
endfor
A(3*n,1) = 1;  # Condición a1=0

# ------ Fill b -------
b(1) = data(1,2);
for j = 1:n-1
    b(2*j) = data(j+1,2);
    b(2*j+1) = data(j+1,2);
endfor
b(2*n) = data(num_points,2);

# ------ Solve the system -------
solucion = linsolve(A,b);
coeficientes = reshape(solucion,n,3)

display(A)
display(b)