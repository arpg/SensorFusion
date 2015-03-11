t = linspace(0,2*pi,40);
syms tsym;
x_func = [cos(tsym) ; sin(tsym) ; tsym*0.2 ; -sin(tsym)*0.5 ; -(tsym+pi/2) ; -sin(tsym)*0.2];
v_func = diff(x_func(1:3),tsym);
w_func = diff(x_func(4:6),tsym);
a_func = diff(v_func,tsym);

 x = cos(t);
 z = sin(t);
 y = t*0.2;
 p = -sin(t)*0.5;
 q = -(t+pi/2);
 r = -sin(t)*0.2;
 plot3(x,y,z)
 axis equal;
 
 
 
 for i = 1:numel(t)
    plot_simple_cf([x(i) y(i) z(i) p(i) q(i) r(i)]',0.1);
 end