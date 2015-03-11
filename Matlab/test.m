function [] = test()
    close all;
    addpath('../../../ThirdParty/mvl/trunk/wrappers/matlab/kinematics');
    addpath('../../../ThirdParty/mvl/trunk/wrappers/matlab/plotutils');

    %trajectory time in seconds
    tf = 5;
    %imu/vicon rate (hz)
    imuRate = 1000;
    viconRate = 100;

    %create a bunch of accelerations, and angular rates
    t = 0:0.05:tf;
    
    %max segments
    maxSegments = 10;


    %calculate T_iv (inertial to vicon reference)
    x_ic = [-0.1 ; 0 ; 0 ; 0 ; 0 ; 0];

    %gravity
    x_ig = [0; 0; 9.81];

    %starting state $6 pose 6 vel 6 bias
    x_0 = zeros(18,1);

    %push forward a dynamics model using these values to obtain inertial
    %and camera poses
    x = [sin(t) ; cos(t)-1 ; cos(t)-1 ; zeros(2,numel(t)) ; cos(t)-1];
    v = [cos(t) ; -sin(t) ; -sin(t) ];

    randMultiplier = 0;
    randOffset = randMultiplier/2;
    w = [zeros(2,numel(t)) ;  -sin(t) + rand(1,numel(t))*randMultiplier - randOffset]; 
    a = [-sin(t)+ rand(1,numel(t))*randMultiplier - randOffset; -cos(t)+ rand(1,numel(t))*randMultiplier - randOffset; -cos(t)+ rand(1,numel(t))*randMultiplier - randOffset + x_ig(3)]; 

    %set initial velocity
    x_0(7:12) = [v(:,1) ; w(:,1)];

    x_cam = [];

    %get the camera poses
    for i = 1:size(x,2)
        x_cam = [x_cam tcomp(x(:,i),x_ic)];
    end


    z = [];

    for i = 1:size(x,2)
        Rwi = Cart2R(x(4:6,i));
        z = [z ; (Rwi'*a(:,i))' w(:,i)'];
    end

    
    errors = [];
    residual = [];
    lastPose = x(:,1);
    x_params = [];

    p_vicon = [];
    p_vicon = x(1:6,1);
    x_params = [[v(1:3,1)' x_ig']];
    x_starting = x(:,1);
    vicon_poses = [t(1) x(1:6,1)'];
    for i=1:maxSegments
        %plot the ground truth
        index_start = 1;
        index_end =(i*10) + 1;

        %plot_simple_path(x_cam(:,index_start:index_end),0.1);

        %now plot the integration
        [rel_pose,txiv,txig] = IntegrateIMU(x(:,1), v(1:3,1), x_ig, [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
        p_vicon = [p_vicon x(1:6,index_end)];
        x_params = [x_params x(1:6,index_end)'];
        lastPose = rel_pose(:,end);
        vicon_poses = [vicon_poses; t(index_end) x(1:6,index_end)'];
        %errors = [errors ; T2Cart(inv(Cart2T(x(:,index_end)))*Cart2T(lastPose))];
    end
    
    %output imu and global poses
    %IMU poses are output as [t ax ay az wx wy wz]
    %vicon poses are output as [t x y z p q r]
    imu_poses = [t' z];
    csvwrite('global_poses.csv',vicon_poses);
    csvwrite('imu_data.csv',imu_poses);
    
    
    %x_params = x_params + rand(size(x_params,1),size(x_params,2))*randMultiplier;
    
    figure
    hold on;
    plot_simple_path(x(:,1:index_end),0.1);
    plot_simple_path(rel_pose,0.1);
    

    %residual = CalculateResidual(x_vicon,x_params, v(1:3,1),x_ig,t,z);
    [pre_residual Jac] = Residual(x_params);
    pre_norm = norm(pre_residual);
    
    results = x_starting;
    for i=2:maxSegments+1
        results = [results x_params((i-1)*6+1:i*6)'];
    end
    %plot_simple_path(results(:,1:end),0.1);
    jacobian = [];
    for k = 1:numel(x_params)
        x_params_new = x_params;
        x_params_new(k) = x_params_new(k) + 0.1;
        plus = Residual(x_params_new);
        x_params_new = x_params;
        x_params_new(k) = x_params_new(k) - 0.1;
        minus = Residual(x_params_new);
        jacobian = [jacobian (plus-minus)/0.2];
    end
    

    %startin guess
    options = optimset('Jacobian', 'on', 'DerivativeCheck', 'on');
    [res,post_norm,post_residual,exitflag,output,lambda,jacobian] = lsqnonlin(@Residual,x_params,-inf,inf,options);
    
    %and now plot the result
    results = x_starting;
    for i=2:maxSegments+1
        results = [results res((i-1)*6+1:i*6)'];
    end
    plot_simple_path(results(:,1:end),0.1);
    
    [rel_pose,txiv,txig] = IntegrateIMU(x_starting, res(1:3)', res(4:6)', [0;0;0;0;0;0], t(1:maxSegments*10 + 1), z(1:maxSegments*10 + 1,:) );
    plot_simple_path(rel_pose,0.05);
    
    i = 0;
    function [r,J] = Residual(params)   
        J = zeros(36,24);
    %         results = [];
    %         for i=2:5
    %             results = [results x(:,i)];
    %         end
    %         plot_simple_path(results(:,1:end),0.1);
        r = [];
        
        for i=1:maxSegments
            %plot the ground truth
            index_start = 1;
            index_end =(i*10) + 1;

            %now plot the integration
            [rel_pose,txiv,txig] = IntegrateIMU(x_starting, params(1:3)', params(4:6)', [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
            Tx = Cart2T(params(i*6+1:(i+1)*6)');
            Ty = inv(Cart2T(rel_pose(:,end)));
            T1 = Tx * Ty;
            r = [r ; T2Cart(T1)];
            Tz = inv(Cart2T(p_vicon(:,i+1)));
            T2 = Tx * Tz;
            r = [r ; T2Cart(T2)];
            
            %calculate the effect of vi and gi
            dt = t(index_end) - t(index_start);
            dGamma = [dt    0   0   -0.5*dt^2    0       0;
                      0     dt  0   0       -0.5*dt^2    0;
                      0     0   dt  0       0       -0.5*dt^2;
                      0     0   0   0       0       0;
                      0     0   0   0       0       0;
                      0     0   0   0       0       0];
            %multiply by the inverse Cart2T transform
%             jf1 = (inv(Cart2T(rel_pose(:,end) + [0.1 0 0 0 0 0]')) - inv(Cart2T(rel_pose(:,end) + [-0.1 0 0 0 0 0]')))/0.2;
%             jf2 = (inv(Cart2T(rel_pose(:,end) + [0 0.1 0 0 0 0]')) - inv(Cart2T(rel_pose(:,end) + [0 -0.1 0 0 0 0]')))/0.2;
%             jf3 = (inv(Cart2T(rel_pose(:,end) + [0 0 0.1 0 0 0]')) - inv(Cart2T(rel_pose(:,end) + [0 0 -0.1 0 0 0]')))/0.2;
%             jf4 = (inv(Cart2T(rel_pose(:,end) + [0 0 0 0.1 0 0]')) - inv(Cart2T(rel_pose(:,end) + [0 0 0 -0.1 0 0]')))/0.2;
%              jf5 = (inv(Cart2T(rel_pose(:,end) + [0 0 0 0 0.1 0]')) - inv(Cart2T(rel_pose(:,end) + [0 0 0 0 -0.1 0]')))/0.2;
%             jf6 = (inv(Cart2T(rel_pose(:,end) + [0 0 0 0 0 0.1]')) - inv(Cart2T(rel_pose(:,end) + [0 0 0 0 0 -0.1]')))/0.2;
            [j1 j2 j3 j4 j5 j6] = dCart2Tinv(rel_pose(:,end));
            jCart2Tinv = [ reshape((Tx*j1)',16,1) reshape((Tx*j2)',16,1) reshape((Tx*j3)',16,1) reshape((Tx*j4)',16,1) reshape((Tx*j5)',16,1) reshape((Tx*j6)',16,1) ];
            jCart2Tinv = jCart2Tinv*dGamma;
            J((i-1)*12+1:(i-1)*12+6,1:6) =   dT2Cart(T1)*jCart2Tinv;
            
%             jf1 = (Cart2T(params(i*6+1:(i+1)*6)' + [0.1 0 0 0 0 0]') - Cart2T(params(i*6+1:(i+1)*6)' + [-0.1 0 0 0 0 0]'))/0.2;
%             jf2 = (Cart2T(params(i*6+1:(i+1)*6)' + [0 0.1 0 0 0 0]') - Cart2T(params(i*6+1:(i+1)*6)' + [0 -0.1 0 0 0 0]'))/0.2;
%             jf3 = (Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0.1 0 0 0]') - Cart2T(params(i*6+1:(i+1)*6)' + [0 0 -0.1 0 0 0]'))/0.2;
%             jf4 = (Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0 0.1 0 0]') - Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0 -0.1 0 0]'))/0.2;
%             jf5 = (Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0 0 0.1 0]') - Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0 0 -0.1 0]'))/0.2;
%             jf6 = (Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0 0 0 0.1]') - Cart2T(params(i*6+1:(i+1)*6)' + [0 0 0 0 0 -0.1]'))/0.2;
            [j1 j2 j3 j4 j5 j6] = dCart2T(params(i*6+1:(i+1)*6)');
            jCart2T = [ reshape((j1*Ty)',16,1) reshape((j2*Ty)',16,1) reshape((j3*Ty)',16,1) reshape((j4*Ty)',16,1) reshape((j5*Ty)',16,1) reshape((j6*Ty)',16,1) ];
            J((i-1)*12+1:(i-1)*12+6,i*6+1:(i+1)*6) =   dT2Cart(T1)*jCart2T;
            
            jCart2T = [ reshape((j1*Tz)',16,1) reshape((j2*Tz)',16,1) reshape((j3*Tz)',16,1) reshape((j4*Tz)',16,1) reshape((j5*Tz)',16,1) reshape((j6*Tz)',16,1) ];
            J((i-1)*12+7:(i-1)*12+12,i*6+1:(i+1)*6) =   dT2Cart(T2)*jCart2T;
            
        end 
        Jac = J;
    end
end

function j = dT2Cart(T)
    j = sparse(6,16);
    j(1,4) = 1.0; %x derivative
    j(2,8) = 1.0; %y derivative
    j(3,12) = 1.0; %z derivative
    j(4,10) = T(3,3)/(T(3,2)^2 + T(3,3)^2); %atan2 -> r22 derivative
    j(4,11) = -T(3,2)/(T(3,2)^2 + T(3,3)^2); %atan2 -> r21 derivative
    j(5,9) = -1/(1 - T(3,1)^2)^(0.5); %asin -> r20 derivative
    j(6,5) = T(1,1)/(T(2,1)^2 + T(1,1)^2); %atan2 -> r10 derivative
    j(6,1) = -T(2,1)/(T(2,1)^2 + T(1,1)^2); %atan2 -> r10 derivative
end

function [j1 j2 j3 j4 j5 j6] = dCart2T(v)
    x = v(1); y = v(2); z = v(3); p = v(4); q = v(5) ; r = v(6);
    j1 = [0 0 0 1;
          0 0 0 0;
          0 0 0 0;
          0 0 0 0];
      
    j2 = [0 0 0 0;
          0 0 0 1;
          0 0 0 0;
          0 0 0 0];
      
    j3 = [0 0 0 0;
          0 0 0 0;
          0 0 0 1;
          0 0 0 0];
        
    j4 = [ 0 sin(p)*sin(r)+cos(p)*cos(r)*sin(q)   cos(p)*sin(r)-cos(r)*sin(p)*sin(q)   0;
           0 cos(p)*sin(q)*sin(r)-cos(r)*sin(p)  -cos(p)*cos(r)-sin(p)*sin(q)*sin(r)   0;
           0                      cos(p)*cos(q)                        -cos(q)*sin(p)  0;
           0                                  0                                     0  0];
                       
    j5 = [ -cos(r)*sin(q)  cos(q)*cos(r)*sin(p) cos(p)*cos(q)*cos(r) 0;
           -sin(q)*sin(r)  cos(q)*sin(p)*sin(r) cos(p)*cos(q)*sin(r) 0;
                 -cos(q)       -sin(p)*sin(q)         -cos(p)*sin(q) 0;
                       0                    0                      0 0];
           
    j6 = [ -cos(q)*sin(r)  -cos(p)*cos(r)-sin(p)*sin(q)*sin(r)  cos(r)*sin(p)-cos(p)*sin(q)*sin(r) 0;
           cos(q)*cos(r)   cos(r)*sin(p)*sin(q)-cos(p)*sin(r)   sin(p)*sin(r)+cos(p)*cos(r)*sin(q) 0;
                       0                                      0                                  0 0;
                       0                                      0                                  0 0];
end

function [j1 j2 j3 j4 j5 j6] = dCart2Tinv(v)
    x = v(1); y = v(2); z = v(3); p = v(4); q = v(5) ; r = v(6);
    j1 = [ 0 0 0                       -cos(q)*cos(r);
           0 0 0   cos(p)*sin(r)-cos(r)*sin(p)*sin(q);
           0 0 0  -sin(p)*sin(r)-cos(p)*cos(r)*sin(q);
           0 0 0                                   0];
      
    j2 = [ 0 0 0                       -cos(q)*sin(r);
           0 0 0  -cos(p)*cos(r)-sin(p)*sin(q)*sin(r);
           0 0 0   cos(r)*sin(p)-cos(p)*sin(q)*sin(r);
           0 0 0                                   0];
      
    j3 = [ 0 0 0         sin(q); 
           0 0 0 -cos(q)*sin(p);
           0 0 0 -cos(p)*cos(q);
           0 0 0              0];
        
    j4 = [                                    0                                    0              0                                                                                             0;
           sin(p)*sin(r) + cos(p)*cos(r)*sin(q)   cos(p)*sin(q)*sin(r)-cos(r)*sin(p)  cos(p)*cos(q) y*(cos(r)*sin(p)-cos(p)*sin(q)*sin(r))-x*(sin(p)*sin(r)+cos(p)*cos(r)*sin(q))-z*cos(p)*cos(q);
           cos(p)*sin(r) - cos(r)*sin(p)*sin(q)  -cos(p)*cos(r)-sin(p)*sin(q)*sin(r) -cos(q)*sin(p) y*(cos(p)*cos(r)+sin(p)*sin(q)*sin(r))-x*(cos(p)*sin(r)-cos(r)*sin(p)*sin(q))+z*cos(q)*sin(p);
                                              0                                    0              0                                                                                             0];

 

                       
    j5 = [       -cos(r)*sin(q)        -sin(q)*sin(r)         -cos(q)                       z*cos(q)+x*cos(r)*sin(q)+y*sin(q)*sin(r);
           cos(q)*cos(r)*sin(p)  cos(q)*sin(p)*sin(r)  -sin(p)*sin(q)  z*sin(p)*sin(q)-x*cos(q)*cos(r)*sin(p)-y*cos(q)*sin(p)*sin(r);
           cos(p)*cos(q)*cos(r)  cos(p)*cos(q)*sin(r)  -cos(p)*sin(q)  z*cos(p)*sin(q)-x*cos(p)*cos(q)*cos(r)-y*cos(p)*cos(q)*sin(r);
                             0                     0                0                                                              0];
 


           
    j6 = [                        -cos(q)*sin(r)                        cos(q)*cos(r) 0                                                       x*cos(q)*sin(r)-y*cos(q)*cos(r);
             -cos(p)*cos(r)-sin(p)*sin(q)*sin(r)   cos(r)*sin(p)*sin(q)-cos(p)*sin(r) 0         x*(cos(p)*cos(r)+sin(p)*sin(q)*sin(r))+y*(cos(p)*sin(r)-cos(r)*sin(p)*sin(q));
              cos(r)*sin(p)-cos(p)*sin(q)*sin(r)   sin(p)*sin(r)+cos(p)*cos(r)*sin(q) 0        -x*(cos(r)*sin(p)-cos(p)*sin(q)*sin(r))-y*(sin(p)*sin(r)+cos(p)*cos(r)*sin(q));
                                               0                                    0 0                                                                                     0];



end


% 
% if 0
%     %calculate the jacobian for the camera calibration
%     JtJ = zeros(6,6);
%     Jb = zeros(6,1);
%     x_ic0 = [0;0;0;0;0;0];  %the starting camera calibrations
%     sigma = 0.1;
% 
%     delta = 1000;
%     while norm(delta) > 1e-4
%         totalb = 0;
%         for i=1:9
%            J = [];
%            segIndex = (i*10) + 1;
%            %calculate the error based on our current estimate
%            b = x_cam(:,segIndex)-tcomp(x(:,segIndex),x_ic0);
%            totalb = totalb + norm(b);
%            for j=1:6
%                %change the calibration params across this axis
%                x_ic0(j) = x_ic0(j) + sigma;
%                ePlus = x_cam(:,segIndex) - tcomp(x(:,segIndex),x_ic0);
%                x_ic0(j) = x_ic0(j) - 2*sigma;
%                eMinus = x_cam(:,segIndex) - tcomp(x(:,segIndex),x_ic0);
%                x_ic0(j) = x_ic0(j) + sigma;
%                J(:,j) = (ePlus-eMinus)/(2*sigma);
%            end
%            JtJ = JtJ + J'*J;
%            Jb = Jb + J'*b;
%         end
% 
%         %and now run GN
%         delta = -inv(JtJ)*Jb;
%         x_ic0 = x_ic0 + delta;
%         x_ic0' 
%         totalb
%     end
% end
%plot the integrated path


% 
% x_cam = [];
% 
% 
% plot_simple_path(x_cam,0.2);




