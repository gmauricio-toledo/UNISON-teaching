clc;

function Y = F(x,y1,y2)
  Y = ...
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
