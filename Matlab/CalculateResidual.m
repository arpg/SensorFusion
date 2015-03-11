%here x is an 6xn matrix of poses (free parameters) (1 column per pose)
%g is a 6xn matrix of global poses (1 column per pose)
%vi is the starting velocity
%gi is the starting gravity
%za is a nx6 matrix of observations (3 accel, 3 gyro) (1 row per obs)
% function [residual] = CalculateResidual(x,p,vi,gi,t,z)
%     residual = []
%     for i=1:10
%         %plot the ground truth
%         index_start = 1;
%         index_end =(i*10) + 1;
%         
%         %now plot the integration
%         [rel_pose,txiv,txig] = IntegrateIMU(p(:,1), vi, gi, [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
%         residual = [residual ; T2Cart(Cart2T(x(:,i+1))' * Cart2T(rel_pose(:,end)))];
%         residual = [residual ; T2Cart(Cart2T(x(:,i+1))' * Cart2T(p(:,end)))];
%         %errors = [errors ; 
%         %          T2Cart(Cart2T(x(:,i+1))' * Cart2T(rel_pose(:,end))) ;
%         %          T2Cart(Cart2T(p(:,i+1))' * Cart2T(rel_pose(:,end)))];
%     end
%     
% end

function [residual] = CalculateResidual(x)
    residual = [];
    for i=1:10
        %plot the ground truth
        index_start = 1;
        index_end =(i*10) + 1;
        
        %now plot the integration
        [rel_pose,txiv,txig] = IntegrateIMU(x(1+6:6+6), x(1:3), x(4:6), [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
        residual = [residual ; T2Cart(Cart2T(x(((i+1)*6)+1:((i+1)*6)+6)')' * Cart2T(rel_pose(:,end)))];
        residual = [residual ; T2Cart(Cart2T(x(((i+1)*6)+1:((i+1)*6)+6)')' * Cart2T(p(:,end)))];
        %errors = [errors ; 
        %          T2Cart(Cart2T(x(:,i+1))' * Cart2T(rel_pose(:,end))) ;
        %          T2Cart(Cart2T(p(:,i+1))' * Cart2T(rel_pose(:,end)))];
    end 
end