#include "SensorFusionCeres.h"
#include "Utils.h"
#include "CostFunctions.h"

static double& g_dImuCauchyNorm = CVarUtils::CreateGetUnsavedCVar("debug.ImuCauchyNorm",0.2);
static double& g_dGlobalCauchyNorm = CVarUtils::CreateGetUnsavedCVar("debug.GlobalCauchyNorm",0.01);


namespace fusion
{
/////////////////////////////////////////////////////////////////////////////////////////
SensorFusionCeres::SensorFusionCeres(const int nFilterSize) :
    m_nFilterSize(nFilterSize),
    m_nInterationNum(1),
    m_dImuTimeOffset(0),
    m_dGlobalTimeOffset(0),
    m_bTimeCalibrated(false),
    m_bCalibActive(false),
    m_bCanAddImu(true),
    m_bFirstPose(false),
    m_EigenFormat(3, 0, ", ", "\n" , "[" , "]"),
    m_dStartTime(-1),
    m_dAccelBias(0,0,0),
    m_dGyroBias(0,0,0)

{

}

/////////////////////////////////////////////////////////////////////////////////////////
void SensorFusionCeres::RegisterImuPose(double accelX, double accelY, double accelZ, double gyroX, double gyroY, double gyroZ, double imuTime, double time)
{
    //add this to the end of the imu data array
    if(m_bTimeCalibrated == true){
        //either add the data to the queue or integrate it immediately
        double correctedTime = imuTime + m_dImuTimeOffset;
        //dout("Imu pose at corrected time " << std::setprecision (15) << correctedTime);
        ImuData data;
        data.m_dAccels = Eigen::Vector3d( accelX,accelY,accelZ);
        data.m_dGyros = Eigen::Vector3d( gyroX,gyroY,gyroZ);
        data.m_dImuTime = imuTime;
        data.m_dTime = correctedTime;


        std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
        if(m_bCanAddImu){
            dout("Integrating forward from " << m_lImuData.back().m_dTime-m_dStartTime << " to " << data.m_dTime-m_dStartTime << " with dt " << data.m_dTime-m_lImuData.back().m_dTime);
            _RegisterImuPoseInternal(data);
        }else{
            dout("Cannot integrate forward. Pushing to queue");
            m_lImuQueue.push_back(data);
        }
    }else if(m_bFirstPose == false){
        if(m_lImuData.size() < REQUIRED_TIME_SAMPLES){
            m_lImuData.push_back(ImuData());
            ImuData& data = m_lImuData.back();
            data.m_dAccels = Eigen::Vector3d( accelX,accelY,accelZ);
            data.m_dGyros = Eigen::Vector3d( gyroX,gyroY,gyroZ);
            data.m_dImuTime = imuTime;
            data.m_dTime = time;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
void SensorFusionCeres::_RegisterImuPoseInternal(const ImuData& data)
{
    if(m_dStartTime == -1){
        m_dStartTime = data.m_dTime;
    }
    m_lImuData.push_back(data);
    //advance the latest pose
    if(m_lParams.size() != 0 && m_lImuData.size() > 1){
        //integrate the IMU one step to push the state forward
        std::list<ImuData>::iterator it = m_lImuData.end();
        ImuData& end = *(--it);   //get the last element
        ImuData& start = *(--it);     //get the one before last element

        //dout("Integrating IMU from  " << std::setprecision (15) << start.m_dTime << " to " << end.m_dTime);
        m_CurrentPose = _IntegrateImuOneStep(m_CurrentPose,start,end,GetGravityVector(m_dG));
        m_CurrentPose.m_dW = data.m_dGyros;
    }
    m_CurrentPose.m_dTime = data.m_dTime;
}

/////////////////////////////////////////////////////////////////////////////////////////
void SensorFusionCeres::RegisterGlobalPose(const Sophus::SE3d& dGlobalPose,
                                           const double& viconTime,
                                           const double& time)
{
    RegisterGlobalPose(dGlobalPose,PoseData(),viconTime,time, true,false);
}

/////////////////////////////////////////////////////////////////////////////////////////
void SensorFusionCeres::RegisterGlobalPose(const Sophus::SE3d& dT_wc,
                                           const PoseData& relativePose,
                                           const double& viconTime,
                                           const double& time,
                                           const bool& bHastGlobalPose,
                                           const bool& bHasRelativePose)
{
    m_bFirstPose = false;
    PoseData data;
    data.m_dSensorTime = viconTime;
    data.m_dPose = dT_wc;

    //make sure no more imu poses are added after this point
    {
        std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
        m_bCanAddImu = false;
    }


    if(m_bTimeCalibrated == false){
        if(m_lGlobalPoses.size() < REQUIRED_TIME_SAMPLES){
            data.m_dTime = time;
            m_lGlobalPoses.push_back(data);
        }else{
            //do time calibration
            _CalibrateTime();
            m_lImuData.clear();
            m_lGlobalPoses.clear();
            m_bTimeCalibrated = true;
            //set the current pose to this global pose
            m_CurrentPose.m_dPose = dT_wc*m_dTic.inverse()*m_dTim;
            m_CurrentPose.m_dV.setZero();
            m_CurrentPose.m_dW.setZero();
        }
    }else{
        double correctedTime = viconTime + m_dGlobalTimeOffset;
        //finds the first pose based on the filter size
        while((int)m_lParams.size() >= m_nFilterSize+1){
            std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
            std::list<ImuData>::iterator it = m_lImuData.begin();
            std::list<ImuData>::iterator prev = it;
            //find the first IMU datapoint (interpolate if necessary)
            double tStart = m_lParams.front().m_dTime;
            while(it != m_lImuData.end() && (*it).m_dTime < tStart) {
                prev = it;
                it++;
            }
            if(prev != it){
                m_lImuData.erase(m_lImuData.begin(),prev);
            }

            //delete a parameter from the beginning
            m_lParams.pop_front();
        }

        //add one to the new poses array
        //TODO: Setting the time to the current time reduces the imu residual, however this is not the right way
        //to do this. Interpolation is causing some errors (maybe)
        data.m_dTime = correctedTime;

        //add a new parameter for this pose at the end
        //but initialize it to the IMU integrated position
        {
            //integrate the previous IMU measurement up to this time
            {
                std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
                if(m_lImuData.size() > 0){
                    ImuData imuData = m_lImuData.back();
                    imuData.m_dTime = data.m_dTime;
                    dout("Integrating forward to match vicon from " << m_lImuData.back().m_dTime-m_dStartTime << " to " << data.m_dTime-m_dStartTime << " with dt " << data.m_dTime-m_lImuData.back().m_dTime );
                    _RegisterImuPoseInternal(imuData);

                }
            }

            //m_CurrentPose.m_dTime = data.m_dTime;
            PoseParameter currentParam = m_CurrentPose;
            currentParam.m_dTime = data.m_dTime;
            //set the global pose
            currentParam.m_GlobalPose = data;
            currentParam.m_bHasGlobalPose = bHastGlobalPose;
            currentParam.m_bHasRelativePose = bHasRelativePose;
            currentParam.m_RelativePose = relativePose;

            m_lParams.push_back(currentParam);

        }

        //recover the zero initial velocity, zero gravity IMU delta for the optimization
        //Eigen::Matrix4d imuDeltaT = Eigen::Matrix4d::Identity();
        //Eigen::Vector3d imuDeltaV = Eigen::Vector3d::Zero();
        if(m_lParams.size() > 1){
            std::list<PoseParameter>::iterator paramIt = m_lParams.end();
            paramIt--;
            PoseParameter& currentParam = *paramIt;
            paramIt--;
            PoseParameter& prevParam = *paramIt;
            double dt = (currentParam.m_dTime - prevParam.m_dTime);

            prevParam.m_dImuDeltaT = currentParam.m_dPose;
            //remove the initial velocity and gravity effect
            Eigen::Vector3d gravity = GetGravityVector(m_dG);
            //augment the transform to make it zero gravity and zero initial velocity
            prevParam.m_dImuDeltaT.translation() -=  (-gravity*0.5*dt*dt + prevParam.m_dV*dt);
            prevParam.m_dImuDeltaT = prevParam.m_dPose.inverse() * prevParam.m_dImuDeltaT;
            //also augment the velocity by subtracting effects of gravity
            prevParam.m_dImuDeltaV = currentParam.m_dV - prevParam.m_dV;
            prevParam.m_dImuDeltaV += gravity*dt;

            //rotate this velocity so that it starts from orientation=Ident
            prevParam.m_dImuDeltaV = prevParam.m_dPose.so3().inverse() * prevParam.m_dImuDeltaV;
        }

        //double norm = DBL_MAX;
        //if(m_lParams.size() >= m_nFilterSize){
        _OptimizePoses();
        //}

        //set the current pose as the last parameter
        {
            m_CurrentPose.m_dPose = m_lParams.back().m_dPose;
            m_CurrentPose.m_dV = m_lParams.back().m_dV;
            m_CurrentPose.m_dW = m_lParams.back().m_dW;
            m_CurrentPose.m_dTime = correctedTime;
        }
    }

    //re enable imu integration
    {
        std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
        //go through the queue and integrate all leftover itmes
        for( std::list<ImuData>::iterator iter = m_lImuQueue.begin() ; iter != m_lImuQueue.end() ; iter++ ){
            //if((*iter).m_dTime > m_CurrentPose.m_dTime){
            dout("Integrating queued imu forward from " << m_lImuData.back().m_dTime-m_dStartTime << " to " << (*iter).m_dTime-m_dStartTime << " with dt " << (*iter).m_dTime-m_lImuData.back().m_dTime);
                _RegisterImuPoseInternal(*iter);
            //}
        }
        m_lImuQueue.clear();
        m_bCanAddImu = true;
    }
}

void SensorFusionCeres::ResetCurrentPose(const Sophus::SE3d& pose, const Eigen::Vector3d& initV, const Eigen::Vector2d& initG)
{
    std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
    m_CurrentPose.m_dPose = pose;
    m_CurrentPose.m_dV = initV;
    m_dG = initG;
    m_dG_information = Eigen::MatrixXd(2,2);
    m_dG_information.setZero();
    m_dPose_information = Eigen::MatrixXd(7,7);
    m_dPose_information.setZero();
    m_dV_information = Eigen::MatrixXd(3,3);
    m_dV_information.setZero();

    m_dTic = Sophus::SE3d();

    //clear all the array
    m_lImuData.clear();
    m_lParams.clear();
    m_lGlobalPoses.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////
double SensorFusionCeres::_OptimizePoses()
{
    ceres::Problem::EvaluateOptions evalOptions;
    Eigen::MatrixXd dG_prev = m_dG;
    double* pPrePriorPose = NULL;
    double* pPrePriorV = NULL;
    const Eigen::Map<const Eigen::Matrix<double,7,1> > dPose_prev(m_lParams.front().m_dPose.data());
    Eigen::Vector3d dV_prev = m_lParams.front().m_dV;

    //std::list<PoseData>::iterator currentModelPose = m_lModelPoses.begin();
    std::list<PoseParameter>::iterator currentParam = m_lParams.begin();
    std::list<PoseParameter>::iterator prevParam = m_lParams.begin();
    int numPoses = m_lParams.size();

    //exit if we don't have enough poses for optimization
    if(numPoses == 0){
        return 0;
    }

    ceres::Problem problem;
    //problem.Evaluate();

    // Special local compositional update for Lie Group SE3
    ceres::LocalParameterization* se3param = new LocalParamSe3();
    problem.AddParameterBlock(m_dTic.data(),7,se3param);
    //problem.AddParameterBlock(m_dTim.data(),7,se3param);

    if(m_bCalibActive == false){
        problem.SetParameterBlockConstant(m_dTic.data());
        //problem.SetParameterBlockConstant(m_dTim.data());
    }

    //go through the poses`
    int paramIndex = 0;
    int weight = 1;
    ceres::ResidualBlockId firstId = NULL;
    while(currentParam !=  m_lParams.end()){
        //enforce the local parametrization
        problem.AddParameterBlock((*currentParam).m_dPose.data(),7,se3param);
        //problem.SetParameterBlockConstant(m_lParams.front().m_dPoseParam.data());


        if((*currentParam).m_bHasGlobalPose == true){
            GlobalPoseCostFunction* pCost = new GlobalPoseCostFunction((*currentParam).m_GlobalPose.m_dPose,weight);
            problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<GlobalPoseCostFunction,6,7,7>(pCost),
                        new ceres::CauchyLoss(g_dGlobalCauchyNorm),m_dTic.data(), (*currentParam).m_dPose.data());
        }

        //add in the error function for the previous pose moving
        if(currentParam != m_lParams.begin()){

            double dt = (*currentParam).m_dTime - (*prevParam).m_dTime;

            ceres::ResidualBlockId id = problem.AddResidualBlock( new ceres::AutoDiffCostFunction<ImuCostFunction,9,7,7,3,3,2>(
                                                new ImuCostFunction((*prevParam).m_dImuDeltaT,
                                                                    (*prevParam).m_dImuDeltaV,dt,weight)
                                                    ),
                                                new ceres::CauchyLoss(g_dImuCauchyNorm),(*currentParam).m_dPose.data(), (*prevParam).m_dPose.data(),
                                                (*currentParam).m_dV.data(),(*prevParam).m_dV.data(),m_dG.data());

            if(firstId == NULL){
                firstId = id;
                pPrePriorPose = (*currentParam).m_dPose.data();
                pPrePriorV = (*currentParam).m_dV.data();
            }

            //dout("Relative pose velocities: " << (*currentParam).m_dV.transpose() << " - "  << (*prevParam).m_dV.transpose());
            if((*currentParam).m_bHasRelativePose == true && (*prevParam).m_bHasRelativePose == true){
                RelativePoseCostFunction* pCost =
                        new RelativePoseCostFunction(
                            (*currentParam).m_RelativePose.m_dPose,
                            (*currentParam).m_RelativePose.m_dV,
                            (*currentParam).m_RelativePose.m_dW,
                            weight);
                problem.AddResidualBlock(
                            new ceres::AutoDiffCostFunction<RelativePoseCostFunction,9,7,7,7,3,3>(
                                pCost),
                            new ceres::CauchyLoss(0.5),
                            m_dTim.data(), (*currentParam).m_dPose.data(), (*prevParam).m_dPose.data(),
                            (*currentParam).m_dV.data(),(*prevParam).m_dV.data());//,
                            //(*currentParam).m_dW.data(),(*prevParam).m_dW.data());
            }
        }

        prevParam = currentParam;

        paramIndex++;
        currentParam++;
        //weight += 0.01;
    }

    //add the prior cost for gravity, prior pose and prior V
    problem.AddResidualBlock( new ceres::AutoDiffCostFunction<PriorCostFunction<2>,2,2>(new PriorCostFunction<2>(dG_prev,m_dG_information)),NULL,m_dG.data());
    //problem.AddResidualBlock( new ceres::AutoDiffCostFunction<PriorCostFunction<7>,7,7>(new PriorCostFunction<7>(dPose_prev,m_dPose_information)),NULL,m_lParams.front().m_dPose.data());
    //problem.AddResidualBlock( new ceres::AutoDiffCostFunction<PriorCostFunction<3>,3,3>(new PriorCostFunction<3>(dV_prev,m_dV_information)),NULL,m_lParams.front().m_dV.data());

    //now solve this problem
    ceres::Solver::Options options;
    options.max_num_iterations = m_bCalibActive || m_lParams.size() < m_nFilterSize ? 10 : m_nInterationNum;
    //options.max_solver_time_in_seconds = 1.0/130.0;
    options.num_threads = 1;
    options.check_gradients = false;
    //options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = false;
    //options.max_num_iterations = 1;
    //options.return_initial_residuals = false;
    //options.return_final_residuals = false;
    //options.max_num_iterations =1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    //evaluate the gravity prior jacobian
    if(firstId != NULL && pPrePriorV != NULL && pPrePriorPose != NULL){
        ceres::CRSMatrix jacobian;
        evalOptions.residual_blocks.push_back(firstId);
        evalOptions.parameter_blocks.push_back(m_dG.data());
        problem.Evaluate(evalOptions,NULL,NULL,NULL,&jacobian);

        //construct the dense jacobian
        Eigen::MatrixXd J = CRSMatrixToEigen(jacobian);
        m_dG_information += J.transpose()*J;


//            evalOptions.parameter_blocks.clear();
//            evalOptions.parameter_blocks.push_back(pPrePriorV);
//            problem.Evaluate(evalOptions,NULL,NULL,NULL,&jacobian);

//            J = CRSMatrixToEigen(jacobian);
//            m_dV_information = J.transpose()*J;

//            evalOptions.parameter_blocks.clear();
//            evalOptions.parameter_blocks.push_back(pPrePriorPose);
//            problem.Evaluate(evalOptions,NULL,NULL,NULL,&jacobian);

//            J = CRSMatrixToEigen(jacobian);
//            double pSe3paramJacobian[100];
//            se3param->ComputeJacobian(pPrePriorPose,pSe3paramJacobian);
//            const Eigen::Map<const Eigen::Matrix<double,6,7> > dSe3ParamJacobian(pSe3paramJacobian);
//            Eigen::MatrixXd JSe3 = J*dSe3ParamJacobian;
//            m_dPose_information = JSe3.transpose()*JSe3;
//        std::cout << "New pose information matrix: " << m_dPose_information << std::endl;

    }


    //std::cout << summary.FullReport() << std::endl;
    m_dRMSE = summary.final_cost;
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////
void SensorFusionCeres::SetCalibrationPose(const Sophus::SE3d& dTic)
{
    m_dTic = dTic;

    Sophus::SE3d T_cm_default(Sophus::SO3d(),Eigen::Vector3d(0,0,0.07));
    m_dTim = m_dTic*T_cm_default;
}

/////////////////////////////////////////////////////////////////////////////////////////
PoseParameter SensorFusionCeres::GetCurrentPose()
{
    std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
    PoseParameter param = m_CurrentPose;
    param.m_dPose = param.m_dPose*m_dTim;
    return param;
}

/////////////////////////////////////////////////////////////////////////////////////////
PoseParameter SensorFusionCeres::_IntegrateImu(const PoseParameter& startingPose, double tStart, double tEnd, Eigen::Vector3d g, std::vector<Sophus::SE3d>* posesOut /*= NULL*/)
{
    std::unique_lock<std::mutex> lock(m_ImuLock, std::try_to_lock);
    PoseParameter startPose = startingPose;
    //go through the IMU values and integrate along
    std::list<ImuData>::iterator it = m_lImuData.begin();
    std::list<ImuData>::iterator prev = it;
    std::list<ImuData>::iterator next = it;

    //calculate gravity
    //calculate the full gravity vector
    //Eigen::Vector3d g = GetGravityVector(m_dG);

    //find the first IMU datapoint (interpolate if necessary)
    while(it != m_lImuData.end() && (*it).m_dTime < tStart) {
        prev = it;
        it++;
    }

    //if we have reached the end m then there is nowhere to go
    if(it == m_lImuData.end()){
        return startPose;
    }

    if(posesOut != NULL){
        posesOut->push_back(startPose.m_dPose);
    }

    ImuData interpolatedData = *it;
    if(prev != it){
        //then we have to interpolate
        double alpha = (tStart - (*prev).m_dTime)  / ((*it).m_dTime - (*prev).m_dTime);
        interpolatedData.m_dAccels = (*it).m_dAccels*alpha + (*prev).m_dAccels*(1-alpha);
        interpolatedData.m_dGyros = (*it).m_dGyros*alpha + (*prev).m_dGyros*(1-alpha);
        interpolatedData.m_dTime = tStart;
        startPose = _IntegrateImuOneStep(startPose,interpolatedData,(*it),g);
        if(posesOut != NULL){
            posesOut->push_back(startPose.m_dPose);
        }
    }



    prev = it;
    it++;
    //and now integrate to the last timestep
    while(it != m_lImuData.end() && (*it).m_dTime < tEnd) {
        startPose = _IntegrateImuOneStep(startPose,(*prev),(*it),g);
        prev = it;
        it++;
        if(posesOut != NULL){
            posesOut->push_back(startPose.m_dPose);
        }
    }

    //if we have reached the end m then there is nowhere to go
    if(it == m_lImuData.end()){
        return startPose;
    }

    //then we will have to interpolate the last step
    interpolatedData = *it;
    if(prev != it){
        //then we have to interpolate
        double alpha = (tEnd - (*prev).m_dTime)  / ((*it).m_dTime - (*prev).m_dTime);
        interpolatedData.m_dAccels = (*it).m_dAccels*alpha + (*prev).m_dAccels*(1-alpha);
        interpolatedData.m_dGyros = (*it).m_dGyros*alpha + (*prev).m_dGyros*(1-alpha);
        interpolatedData.m_dTime = tEnd;
        startPose = _IntegrateImuOneStep(startPose,(*prev),interpolatedData,g);
        if(posesOut != NULL){
            posesOut->push_back(startPose.m_dPose);
        }
    }
    return startPose;
}

/////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
Eigen::Matrix<T,9,1> SensorFusionCeres::_GetPoseDerivative(const Sophus::SE3Group<T>& tTwb,const Eigen::Matrix<T,3,1>& tV_w,  const Eigen::Matrix<T,3,1>& tG_w, const ImuDataBase<T>& zStart, const ImuDataBase<T>& zEnd, const T dt)
{
    T alpha = (zEnd.m_dTime - (zStart.m_dTime+dt))/(zEnd.m_dTime - zStart.m_dTime);
    Eigen::Matrix<T,3,1> zb = zStart.m_dGyros*alpha + zEnd.m_dGyros*((T)1-alpha);
    Eigen::Matrix<T,3,1> za = zStart.m_dAccels*alpha + zEnd.m_dAccels*((T)1-alpha);

    Eigen::Matrix<T,9,1> deriv;
    //Eigen::Matrix<T,3,1> zw_world = br2arT * zw;
    //derivative of position is velocity
    deriv.head(3) = tV_w;
    //deriv.template segment<3>(3) = Sophus::SO3Group<T>::vee(tTwb.so3().matrix()*Sophus::SO3Group<T>::hat(zb));
    deriv.template segment<3>(3) = tTwb.so3().Adj()*zb;
    deriv.template segment<3>(6) = tTwb.so3()*(za) - tG_w;
    return deriv;
}

/////////////////////////////////////////////////////////////////////////////////////////
//    template <typename T>
//    Eigen::Matrix<T,9,1> SensorFusionCeres::_GetPoseDerivative(Eigen::Matrix<T,9,1> dState, Eigen::Matrix<T,3,1> dG, const ImuDataBase<T>& zStart, const ImuDataBase<T>& zEnd, const T dt)
//    {
//        T sp = sin(dState[3]); // roll phi
//        T cp = cos(dState[3]); // roll
//        T cq = cos(dState[4]); // pitch
//        T sq = sin(dState[4]); // pitch
//        T tq = tan(dState[4]); // pitch
//        T cr = cos(dState[5]); // yaw
//        T sr = sin(dState[5]); // yaw
//        //T tr = tan(dState[5]); // yaw

//        //matrix from inertial to world coordinates
//        Eigen::Matrix<T,3,3> Rwi;// = mvl::Cart2R(dState.block<3,1>(3,0));

//        Rwi(0,0) = cq*cr;
//        Rwi(0,1) = -cp*sr+sp*sq*cr;
//        Rwi(0,2) = sp*sr+cp*sq*cr;

//        Rwi(1,0) = cq*sr;
//        Rwi(1,1) = cp*cr+sp*sq*sr;
//        Rwi(1,2) = -sp*cr+cp*sq*sr;

//        Rwi(2,0) = -sq;
//        Rwi(2,1) = sp*cq;
//        Rwi(2,2) = cp*cq;

//        //conversion from body rates to angular rate
//        Eigen::Matrix<T,3,3> br2arT;
//        br2arT <<   (T)1,   sp*tq,   cp*tq,
//                    (T)0,      cp,     -sp,
//                    (T)0,   sp/cq,   cp/cq;

//        T alpha = (zEnd.m_dTime - (zStart.m_dTime+dt))/(zEnd.m_dTime - zStart.m_dTime);
//        Eigen::Matrix<T,3,1> zw = zStart.m_dGyros*alpha + zEnd.m_dGyros*((T)1-alpha);
//        Eigen::Matrix<T,3,1> za = zStart.m_dAccels*alpha + zEnd.m_dAccels*((T)1-alpha);


//        Eigen::Matrix<T,9,1> deriv;
//        Eigen::Matrix<T,3,1> zw_world = br2arT * zw;
//        //derivative of position is velocity
//        deriv.head(3) = dState.template block<3,1>(6,0);
//        deriv.template block<3,1>(3,0) = zw_world;

//        deriv.template block<3,1>(6,0) = Rwi*(za) - dG;
//        return deriv;
//    }

/////////////////////////////////////////////////////////////////////////////////////////
void SensorFusionCeres::_CalibrateTime()
{
    double totalImuDifference = 0;
    for( std::list<ImuData>::iterator iter = m_lImuData.begin() ; iter != m_lImuData.end() ; iter++ ){
        totalImuDifference += (*iter).m_dTime - (*iter).m_dImuTime;
    }
    m_dImuTimeOffset = totalImuDifference / m_lImuData.size();



    double totalGlobalDifference = 0;
    for( std::list<PoseData>::iterator iter = m_lGlobalPoses.begin() ; iter != m_lGlobalPoses.end() ; iter++ ){
        totalGlobalDifference += (*iter).m_dTime - (*iter).m_dSensorTime;
    }
    m_dGlobalTimeOffset = totalGlobalDifference / m_lGlobalPoses.size();

}

/////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
Eigen::Matrix<T,3,1> SensorFusionCeres::GetGravityVector(const Eigen::Matrix<T,2,1> &direction)
{
    T sp = sin(direction[0]);
    T cp = cos(direction[0]);
    T sq = sin(direction[1]);
    T cq = cos(direction[1]);
    Eigen::Matrix<T,3,1> g(cp*sq,-sp,cp*cq);
    g *= -(T)IMU_GRAVITY_CONST;
    return g;
}

/////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3d SensorFusionCeres::GetGravityVector(const Eigen::Vector2d &direction)
{
    return GetGravityVector<double>(direction);
}

/////////////////////////////////////////////////////////////////////////////////////////
PoseParameter SensorFusionCeres:: _IntegrateImuOneStep(const PoseParameter& currentPose, const ImuData& zStart, const ImuData &zEnd, const Eigen::Vector3d dG)
{
    return _IntegrateImuOneStepBase<double>(currentPose,zStart,zEnd,dG);
}

//    template <typename T>
//    PoseParameterBase<T> SensorFusionCeres::_IntegrateImuOneStepBase(const PoseParameterBase<T>& currentPose, const ImuDataBase<T>& zStart, const ImuDataBase<T> &zEnd, const Eigen::Matrix<T,3,1> dG)
//    {
//        //construct the state matrix
//        Eigen::Matrix<T,9,1> dState;
//        dState.head(6) = mvl::T2Cart(currentPose.m_dPose.matrix());
//        dState.tail(3) = currentPose.m_dV;
//        T h = zEnd.m_dTime - zStart.m_dTime;
//        if(h == (T)0){
//            return currentPose;
//        }
//        Eigen::Matrix<T,9,1> k1 = _GetPoseDerivative<T>(dState,dG,zStart,zEnd,(T)0);
//        Eigen::Matrix<T,9,1> k2 = _GetPoseDerivative<T>(dState + (T)0.5*h*k1,dG,zStart,zEnd,(T)0.5*h);
//        Eigen::Matrix<T,9,1> k3 = _GetPoseDerivative<T>(dState + (T)0.5*h*k2,dG,zStart,zEnd,(T)0.5*h);
//        Eigen::Matrix<T,9,1> k4 = _GetPoseDerivative<T>(dState + h*k3,dG,zStart,zEnd,h);
//        dState = dState + ((T)1.0/(T)6.0)*h*(k1 + (T)2*k2 + (T)2*k3 + k4);


//        //and now output the state
//        PoseParameterBase<T> output;
//        output.m_dPose = Sophus::SE3d(mvl::Cart2T(dState.head(6)));
//        output.m_dV = dState.tail(3);
//        output.m_dW = currentPose.m_dW;
//        output.m_dTime = zEnd.m_dTime;
//        return output;
//    }

/////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
PoseParameterBase<T> SensorFusionCeres::_IntegrateImuOneStepBase(const PoseParameterBase<T>& currentPose, const ImuDataBase<T>& zStart, const ImuDataBase<T> &zEnd, const Eigen::Matrix<T,3,1> dG)
{
    //construct the state matrix
    Eigen::Matrix<T,9,1> dState;
    dState.head(6) = fusion::T2Cart(currentPose.m_dPose.matrix());
    dState.tail(3) = currentPose.m_dV;
    T h = zEnd.m_dTime - zStart.m_dTime;
    if(h == (T)0){
        return currentPose;
    }
    Sophus::SE3Group<T> aug_Twv = currentPose.m_dPose;
    Eigen::Matrix<T,3,1> aug_V = currentPose.m_dV;
    Eigen::Matrix<T,9,1> k1 = _GetPoseDerivative<T>(aug_Twv,aug_V,dG,zStart,zEnd,(T)0);

    aug_Twv = currentPose.m_dPose;
    aug_Twv.translation() += k1.head(3)*h;
    Sophus::SO3Group<T> Rv2v1(Sophus::SO3Group<T>::exp(k1.segment(3,3)*h));
    aug_Twv.so3() = Rv2v1*currentPose.m_dPose.so3();

    aug_V += k1.tail(3)*h;
//        Eigen::Matrix<T,9,1> k2 = _GetPoseDerivative<T>(dState + (T)0.5*h*k1,dG,zStart,zEnd,(T)0.5*h);
//        Eigen::Matrix<T,9,1> k3 = _GetPoseDerivative<T>(dState + (T)0.5*h*k2,dG,zStart,zEnd,(T)0.5*h);
//        Eigen::Matrix<T,9,1> k4 = _GetPoseDerivative<T>(dState + h*k3,dG,zStart,zEnd,h);
//        dState = dState + ((T)1.0/(T)6.0)*h*(k1 + (T)2*k2 + (T)2*k3 + k4);


    //and now output the state
    PoseParameterBase<T> output;
    output.m_dPose = aug_Twv;
    output.m_dV = aug_V;
    output.m_dW = currentPose.m_dW;
    output.m_dTime = zEnd.m_dTime;
    return output;
}

}
