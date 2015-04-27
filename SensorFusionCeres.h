#ifndef SensorFusionCeres_H
#define SensorFusionCeres_H
#include <stdio.h>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <math.h>
#include <thread>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "SE3.h"
#include "LocalParamSe3.h"
#include "Utils.h"
#include <CVars/CVar.h>
#include "float.h"



//static bool& g_bImuIntegrationOnly = CVarUtils::CreateUnsavedCVar("debug.ImuIntegrationOnly",false);

#define DEBUG 1
#ifdef DEBUG
#define dout(str) std::cout << __FUNCTION__ << " --  " << str << std::endl
#else
#define dout(str)
#endif


namespace Eigen {
    typedef Matrix<double,7,1> Vector7d ;
typedef Matrix<double,6,1> Vector6d ;
}

namespace fusion
{

    #define ERROR_TERMS_PER_PARAMETER 12
    #define POSE_TERMS 6
    #define PARAM_TERMS 6
    #define INITIAL_VEL_TERMS 3
    #define INITIAL_ACCEL_TERMS 2
    #define IMU_GRAVITY_CONST 9.80665
    #define G_ACCEL 9.80665
    #define REQUIRED_TIME_SAMPLES 100

    typedef std::vector<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>,Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > > VectorXdRowMajAlignedVec;

    template <typename T>  struct ImuDataBase
    {
        Eigen::Matrix<T,3,1> m_dAccels;
        Eigen::Matrix<T,3,1> m_dGyros;
        T m_dImuTime;
        T m_dTime;
    };

    template <typename T>
    struct PoseDataBase
    {
        Sophus::SE3Group<T> m_dPose;
        Eigen::Vector3d m_dV;
        Eigen::Vector3d m_dW;
        T m_dSensorTime;
        T m_dTime;
    };

    template <typename T>
    struct PoseParameterBase
    {
        Sophus::SE3Group<T> m_dPose;
        Eigen::Matrix<T,3,1> m_dV;
        Eigen::Matrix<T,3,1> m_dW;

        double m_dTime;

        Sophus::SE3Group<T> m_dImuDeltaT;
        Eigen::Matrix<T,3,1> m_dImuDeltaV;

        //global pose
        PoseDataBase<T> m_GlobalPose;
        PoseDataBase<T> m_RelativePose;

        //used to store the optimization delta
        //Eigen::Matrix<T,7,1> m_dPoseParam;

        static PoseParameterBase<T> Zero(){
            PoseParameterBase<T> p;
            p.m_dPose.setZero();
            p.m_dV.setZero();
            p.m_dW.setZero();
            p.m_dTime = (T)0;
            Eigen::Map<Sophus::SE3d> poseParam(p.m_dPoseParam.data());
            poseParam = Sophus::SE3d();
        }

        bool m_bHasGlobalPose;
        bool m_bHasRelativePose;
    };

    typedef ImuDataBase<double> ImuData;
    typedef PoseParameterBase<double> PoseParameter;

    struct ImuIntPoseData
    {
        Eigen::Vector6d m_dPose;
        Eigen::Vector3d m_dV;
        Eigen::Vector3d m_dW;
        double m_dTime;
    };

    inline Eigen::MatrixXd CRSMatrixToEigen(const ceres::CRSMatrix& mat)
    {
        Eigen::MatrixXd dMat(mat.num_rows,mat.num_cols);
        dMat.setZero();
        for(size_t row = 0; row < mat.rows.size()-1 ; row++){
            for(int jj = mat.rows[row] ; jj < mat.rows[row+1] ; jj++){
                dMat(row,mat.cols[jj]) = mat.values[jj];
            }
        }
        return dMat;
    }


    typedef PoseDataBase<double> PoseData;

    class SensorFusionCeres
    {

    public:

        SensorFusionCeres(const int nFilterSize);
        void RegisterImuPose(double accelX, double accelY, double accelZ, double gyroX, double gyroY, double gyroZ,  double imuTime, double time);
        void RegisterGlobalPose(const Sophus::SE3d& dGlobalPose,
                                const double& viconTime,
                                const double& time);
        void RegisterGlobalPose(const Sophus::SE3d& dT_wc,
                                const PoseData& relativePose,
                                const double& viconTime,
                                const double& time,
                                const bool& bHastGlobalPose,
                                const bool& bHasRelativePose);
        PoseParameter GetCurrentPose();
        Sophus::SE3d GetCalibrationPose() { return m_dTic; }
        void SetCalibrationPose(const Sophus::SE3d &dTic);
        void SetCalibrationActive(const bool bActive) { m_bCalibActive = bActive; }
        void SetTimeCalibrationActive(const bool bActive) { m_bTimeCalibrated = !bActive; }
        PoseData GetLastGlobalPose() { return m_lParams.back().m_GlobalPose; }
        void ResetCurrentPose(const Sophus::SE3d &pose, const Eigen::Vector3d &initV, const Eigen::Vector2d &initG);
        template <typename T> static Eigen::Matrix<T,3,1> GetGravityVector(const Eigen::Matrix<T,2,1> &direction);
        static Eigen::Vector3d GetGravityVector(const Eigen::Vector2d &direction);
        int* GetFilterSizePtr() { return &m_nFilterSize; }
        int* GetIterationNumPtr() { return &m_nInterationNum; }
        void SetFilterSize(const int nSize){ m_nFilterSize = nSize; }
        double* GetRMSEPtr(){ return &m_dRMSE; }
        double GetGlobalTimeOffset() const { return m_dGlobalTimeOffset; }
        double GetImuTimeOffset() const { return m_dImuTimeOffset; }
        bool IsActive() const { return m_bTimeCalibrated; }
    //private:


        PoseParameter _IntegrateImu(const PoseParameter& startingPose, double tStart, double tEnd, Eigen::Vector3d g, std::vector<Sophus::SE3d> *posesOut = NULL);
        template <typename T>  PoseParameterBase<T> _IntegrateImuOneStepBase(const PoseParameterBase<T>& currentPose, const ImuDataBase<T>& zStart, const ImuDataBase<T> &zEnd, const Eigen::Matrix<T,3,1> dG);
        PoseParameter _IntegrateImuOneStep(const PoseParameter& currentPose, const ImuData& zStart, const ImuData &zEnd, const Eigen::Vector3d dG);
        //template <typename T> Eigen::Matrix<T,9,1> _GetPoseDerivative(Eigen::Matrix<T,9,1> dState, Eigen::Matrix<T,3,1> dG, const ImuDataBase<T>& zStart, const ImuDataBase<T>& zEnd, const T dt);
        template <typename T> Eigen::Matrix<T,9,1> _GetPoseDerivative(const Sophus::SE3Group<T>& tTwv,const Eigen::Matrix<T,3,1>& tV_w,  const Eigen::Matrix<T,3,1>& tG_w, const ImuDataBase<T>& zStart, const ImuDataBase<T>& zEnd, const T dt);
        void _CalibrateTime();
        ImuIntPoseData _GetInterpolatedImuIntegrationPose(double timeStamp);
        void _RegisterImuPoseInternal(const ImuData& data);


        double _OptimizePoses();
        std::list<ImuData> m_lImuData;
        std::list<ImuData> m_lImuQueue;
        std::list<PoseData> m_lGlobalPoses;
        std::list<PoseData> m_lRelativePoses;
        std::list<PoseData> m_lModelPoses;
        std::list<PoseParameter>  m_lParams;

        Sophus::SE3d m_dTic;
        Sophus::SE3d m_dTim;
        Eigen::Vector2d m_dG;
        Eigen::MatrixXd m_dG_information;
        Eigen::MatrixXd m_dPose_information;
        Eigen::MatrixXd m_dV_information;        
        PoseParameter m_CurrentPose;


        int m_nFilterSize;
        int m_nInterationNum;

        //time offset parameters for vicon and imu
        double m_dRMSE;
        double m_dImuTimeOffset;
        double m_dGlobalTimeOffset;
        bool m_bTimeCalibrated;
        bool m_bCalibActive;
        bool m_bCanAddImu;
        bool m_bFirstPose;

        std::mutex m_ImuLock;
        Eigen::IOFormat m_EigenFormat;

        double m_dStartTime;
        Eigen::Vector3d m_dAccelBias;
        Eigen::Vector3d m_dGyroBias;

    };







} // end namespace fusion
#endif // SensorFusionCeres_H
