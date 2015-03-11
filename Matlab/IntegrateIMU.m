% x0   6d pose vector [x y z roll pitch yaw]
% x0v  3d velocity vector
% x0g  2d gravity vector
% xb0  6d bias vector
% t    nd time vector
% z    nx6 accelerometer-gyro measurements

function [poses, final_v, final_g] = IntegrateIMU(x0, x0v, x0g, xb0, t, z )
    xi = [x0; x0v; xb0 ];
    g = xi;
    for ii = 1:size(z,1)-1
        [xf] = IntegrateOneStep( xi, t(ii), t(ii+1), x0g, xb0, z(ii,:)', z(ii+1,:)' );
        g = [g xf]; xi = xf;
    end
    R12 = Cart2R(g(4:6,end));
    final_v = g(7:9,end);
    final_g = x0g;
    %final_g = R12'*x0g;
    %final_v = R12'*g(7:9,end);
    poses = g(1:6,:);
end

function [xf] = IntegrateOneStep( xi, ti, tf, g0, b0, zi, zf )
    function [xdot] = xvec_ode(t,x)
        Rwi = Cart2R( x(4:6) );
        xdot(1:3,1) = x(7:9,1);
        alpha = (tf-t)/(tf-ti);
        zw = zi(4:6)*alpha + zf(4:6)*(1-alpha) + b0(4:6);
        za = zi(1:3)*alpha + zf(1:3)*(1-alpha) + b0(1:3);
        xdot(4:6,1) = BodyRates2EulerRates( x(4:6), zw );
        xdot(7:9,1) = Rwi*za - g0;
        xdot(10:15) = 0;
    end

    function xf = ProcessModel( xi )
        if abs(tf - ti) < 1e-13
            xf = xi;
            return
        end
        [t, xf] = ODE45SingleStep( @xvec_ode, [ti, tf], xi );
    end

    xf = ProcessModel( xi );
end

% Implementation of fourth order Runge-Kutta
function [tf, ft] = ODE45SingleStep( ode_func, t, y0 )
    ti = t(1);
    tf = t(2);
    h = tf - ti;
    k1 = feval( ode_func, ti , y0 );
    k2 = feval( ode_func, ti + 0.5*h, y0 + 0.5*h*k1 );
    k3 = feval( ode_func, ti + 0.5*h, y0 + 0.5*h*k2 );
    k4 = feval( ode_func, ti + h, y0 + h*k3 );
    ft = y0 + 1/6 * h * (k1 + 2*k2 + 2*k3 + k4);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = Cart2R(x)

    R = zeros(3);
    
    r = x(1);
    p = x(2);
    q = x(3);
    
    cr = cos( r );
    cp = cos( p );
    cq = cos( q );

    sr = sin( r );
    sp = sin( p );
    sq = sin( q );

    R(1,1) = cp*cq;
    R(1,2) = -cr*sq+sr*sp*cq;
    R(1,3) = sr*sq+cr*sp*cq;

    R(2,1) = cp*sq;
    R(2,2) = cr*cq+sr*sp*sq;
    R(2,3) = -sr*cq+cr*sp*sq;

    R(3,1) = -sp;
    R(3,2) = sr*cp;
    R(3,3) = cr*cp;
end
    