syms x y z p q r
syms t00 t01 t02 t03 t10 t11 t12 t13 t20 t21 t22 t23 t30 t31 t32 t33
C2T = [cos(r)*cos(q)  -sin(r)*cos(p)+cos(r)*sin(q)*sin(p)   sin(r)*sin(p)+cos(r)*sin(q)*cos(p)  x;
       sin(r)*cos(q)   cos(r)*cos(p)+sin(r)*sin(q)*sin(p)  -cos(r)*sin(p)+sin(r)*sin(q)*cos(p)  y;
       -sin(q)                       cos(q)*sin(p)                       cos(q)*cos(p)        z;
       0             0                                   0                                     1];

C2Tinv = C2T;
C2Tinv(1:3,1:3) = C2T(1:3,1:3).';
C2Tinv(1:3,4) = -C2T(1:3,1:3).' * C2T(1:3,4);

      
vecT2 = reshape(C2T.',16,1);   
%vecT = [t00 t01 t02 t03 t10 t11 t12 t13 t20 t21 t22 t23 t30 t31 t32 t33];
vecT = [C2T(1,1) C2T(1,2) C2T(1,3) C2T(1,4) C2T(2,1) C2T(2,2) C2T(2,3) C2T(2,4) C2T(3,1) C2T(3,2) C2T(3,3) C2T(3,4) C2T(4,1) C2T(4,2) C2T(4,3) C2T(4,4)].';
T2C = [vecT(4);
       vecT(8);
       vecT(12);
       atan(vecT(10)/vecT(11));
       -asin(vecT(9));
       atan(vecT(5)/vecT(1))];