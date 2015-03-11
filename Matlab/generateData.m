close all;
addpath('/Users/nimski/Code/ThirdParty/mvl/wrappers/matlab/kinematics');
addpath('/Users/nimski/Code/ThirdParty/mvl/wrappers/matlab/plotutils');

%trajectory time in seconds
tf = 2*pi;
%imu/vicon rate (hz)
imuRate = 50;
viconRate = 5;

syms tsym;
x_func = [cos(tsym) ; tsym*0.2 ; sin(tsym) ; sin(tsym)*0.2 ; -(tsym+pi/2) ; -sin(tsym)*0.2];
v_func = diff(x_func(1:3),tsym);
w_func = diff(x_func(4:6),tsym);
a_func = diff(v_func,tsym);

%max segments
maxSegments = floor(tf/(1/viconRate));
dt = 1/imuRate;
segmentLength = floor((tf/maxSegments)/dt);


%create a bunch of accelerations, and angular rates
t = 0:dt:tf;





%calculate T_iv (inertial to vicon reference)
x_ic = [0 ; 0 ; 0 ; 0 ; 0 ; 0];
%x_ic = [0.1 ; 0.05 ; -0.05 ; pi/2 ; 3*pi/2 ; pi/4];

%gravity
x_ig = [0; 0; 9.80665];

%starting state $6 pose 6 vel 6 bias
x_0 = zeros(18,1);

x = [];
v = [];
w = [];
a = [];
%push forward a dynamics model using these values to obtain inertial
%and camera poses

for i = 1:numel(t)
    x = [x subs(x_func,tsym,t(i))];
    v = [v subs(v_func,tsym,t(i))];
    w = [w subs(w_func,tsym,t(i))];
    a = [a subs(a_func,tsym,t(i))];
end
a(3,:) = a(3,:) - x_ig(3);

randMultiplier = 0;%0.005;
w = w + rand(3,size(w,2))*randMultiplier*2;
a = a + rand(3,size(w,2))*randMultiplier*2;

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
    w(:,i) = EulerRates2BodyRates( x(4:6,i), w(:,i));
    z = [z ; (Rwi'*a(:,i))' w(:,i)'];
end


errors = [];
residual = [];
lastPose = x(:,1);


x_starting = x(:,1);
vicon_poses = [t(1) t(1) x(1:6,1)'];
rel_pose = [];
rel_pose_segment = x(:,1);
txiv = v(1:3,1);

%[rel_pose,txiv,txig] = IntegrateIMU(x(:,1), v(1:3,1), x_ig, [0;0;0;0;0;0], t(1:end), z(1:end,:) );
    
 for i=1:maxSegments
%     %plot the ground truth
    index_start = ((i-1)*segmentLength) + 1;
    index_end =(i*segmentLength) + 1;

    dt = t(index_end) - t(index_start);
%     
%     
% 
     vicon_poses = [vicon_poses; t(index_end) t(index_end) x_cam(1:6,index_end)'];
% 
%     %now plot the integration
%     %[rel_pose_segment,txiv,txig] = IntegrateIMU(x(:,index_start), v(1:3,index_start), x_ig, [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
%     [rel_pose_segment,txiv,txig] = IntegrateIMU(rel_pose_segment(:,end), txiv(1:3,end), x_ig, [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
%     %test stuff
%     %[rel_pose_segment2,txiv2,txig2] = IntegrateIMU([0 0 0 0 0 0]', [0 0 0]', [0 0 0]', [0;0;0;0;0;0], t(index_start:index_end), z(index_start:index_end,:) );
%     
% %    PrevT = rel_pose_segment(:,end);
% %    PrevT(1:3) = PrevT(1:3) + x_ig*0.5*dt*dt - v(1:3,index_start)*dt;
% %    deltaT = T2Cart(Cart2T(x(:,index_start))^-1 * Cart2T(PrevT));
% 
%     
%     
% %     endVel = v(1:3,index_start) + Cart2R(x(4:6,index_start))*txiv2 - x_ig*dt;
% %     T = Cart2T(rel_pose_segment2(:,end));
% %     endPos = Cart2T(x(:,index_start))*T;
% %     
% %     endPos(1:3,4) = endPos(1:3,4) - x_ig*0.5*dt*dt + v(1:3,index_start)*dt;
% %     act_endPos = Cart2T(rel_pose_segment(:,end));
% 
%     
%     rel_pose = [rel_pose rel_pose_segment];
%     lastPose = rel_pose(:,end);
 end

%add noise to vicon measurements
vicon_poses(3:end,:) = vicon_poses(3:end,:) + rand(size(vicon_poses(3:end,:)))*randMultiplier;

%output imu and global poses
%IMU poses are output as [t ax ay az wx wy wz]
%vicon poses are output as [t x y z p q r]
imu_poses = [t' t' z];
csvwrite('global_poses.csv',vicon_poses);
csvwrite('imu_data.csv',imu_poses);
csvwrite('ground_truth.csv', [t' x']);
csvwrite('ground_truth_vels.csv', [t' v']);

%x_params = x_params + rand(size(x_params,1),size(x_params,2))*randMultiplier;

figure
hold on;
plot_simple_path(x,0.01);
%plot_simple_path(rel_pose,0.05);
plot_simple_path(vicon_poses(:,3:end)',0.1);

return