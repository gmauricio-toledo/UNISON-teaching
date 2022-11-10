clc;
global data = [1,0;4,1;6,7;6.5,8]; # Definimos los datos

function prod = lagrange_n(i,x,grado)
  global data;
  prod = 1;
  for j = 1:(grado+1)
    if j!=i
      prod = prod*((x-data(j,1))/(data(i,1)-data(j,1)));
    endif
  endfor
endfunction

function sum = evaluar(x,grado)
  global data;
  sum = 0;
  for i = 1:(grado+1)
    sum = sum + (lagrange_n(i,x,grado)*data(i,2));
  endfor
endfunction

printf("Datos:")
display(data)
x = 6.2;
y = evaluar(x,3);
printf("La interpolación en x=%d es %d\n", x,y)
