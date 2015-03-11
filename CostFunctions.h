#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include "Eigen/Eigen"
#include "CVars/CVar.h"
#include "sophus/se3.hpp"
#include "SensorFusionCeres.h"

namespace fusion
{

static double& g_dImuTranslationWeight = CVarUtils::CreateGetUnsavedCVar("debug.ImuTranslationWeight",0.1);
static double& g_dImuOrientationWeight = CVarUtils::CreateGetUnsavedCVar("debug.ImuOrientationWeight",0.1);
static double& g_dImuVelocityWeight = CVarUtils::CreateGetUnsavedCVar("debug.ImuVelocityWeight",0.2);


static double& g_dGlobalTranslationWeight = CVarUtils::CreateGetUnsavedCVar("debug.GlobalTranslationWeight",0.1);
static double& g_dGlobalOrientationWeight = CVarUtils::CreateGetUnsavedCVar("debug.GlobalOrientationWeight",0.1);


/// Ceres autodifferentiatable cost function for pose errors.
/// The parameters are error state and should be a 6d pose delta
struct GlobalPoseCostFunction
{
    GlobalPoseCostFunction(const Sophus::SE3d& dMeasurement, const double dWeight  )
    : m_Tw_c(dMeasurement),
      m_dWeight(dWeight)
    {}

    template <typename T>
    bool operator()(const T* const _tTic,const T* const _tTwi, T* residuals) const
    {
        //const Eigen::Map<const Sophus::SE3Group<T> > T_wx(_t); //the pose delta
        const Eigen::Map<const Sophus::SE3Group<T> > T_i_c(_tTic);
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_i(_tTwi);
       Eigen::Map<Eigen::Matrix<T,6,1> > pose_residuals(residuals); //the pose residuals

       pose_residuals = (T_w_i* T_i_c * m_Tw_c.inverse().cast<T>()).log()  * (T)m_dWeight;
       pose_residuals.tail(3) *= (T)g_dGlobalOrientationWeight;
       pose_residuals.head(3) *= (T)g_dGlobalTranslationWeight;
        //pose_residuals.head(3) *= (T)0.9;
        //pose_residuals.tail(4) *= (T)0.3;
        return true;
    }


    const Sophus::SE3d m_Tw_c;
    const double m_dWeight;
};

/// Ceres autodifferentiatable cost function for pose errors.
/// The parameters are error state and should be a 6d pose deltaz
struct RelativePoseCostFunction
{
    RelativePoseCostFunction(const Sophus::SE3d& dTm2m1, const Eigen::Vector3d& dDeltaV, const Eigen::Vector3d& dDeltaW, const double dWeight  )   :
        m_Tm2m1(dTm2m1),
        m_dDeltaV(dDeltaV),
        m_dDeltaW(dDeltaW),
        m_dWeight(dWeight)
    {
    }

    template <typename T>
    bool operator()(const T* const _tTim,const T* const _tTwi2, const T* const _tTwi1,
                    const T* const _tV2, const T* const _tV1,
                    //const T* const _tW2, const T* const _tW1,
                    T* residuals) const
    {
        //the residual vector consists of a 6d pose, a 3d velocity residual and a 3d w residual
        Eigen::Map<Eigen::Matrix<T,6,1> > pose_residuals(residuals); //the pose residuals
        Eigen::Map<Eigen::Matrix<T,3,1> > vel_residuals(&residuals[6]); //the velocity residuals
        //Eigen::Map<Eigen::Matrix<T,3,1> > w_residuals(&residuals[9]); //the velocity residual

        //const Eigen::Map<const Sophus::SE3Group<T> > T_wx(_t); //the pose delta
        const Eigen::Map<const Sophus::SE3Group<T> > T_i_m(_tTim);
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_i1(_tTwi1);
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_i2(_tTwi2);
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_v2(_tV2);
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_v1(_tV1);
        //const Eigen::Map<const Eigen::Matrix<T,3,1> > v_w2(_tW2);
        //const Eigen::Map<const Eigen::Matrix<T,3,1> > v_w1(_tW1);

       pose_residuals = (T_w_i2* T_i_m * m_Tm2m1.cast<T>() * (T_i_m).inverse() * (T_w_i1).inverse()).log();
       vel_residuals = (v_v2 - (m_dDeltaV.cast<T>() + v_v1));//  * (T)(100/m_dT);
       //w_residuals = (v_w2 - (m_dDeltaW.cast<T>()+ v_w1)) * (T)0;//  * (T)(100/m_dT);
        return true;
    }

    const Sophus::SE3d m_Tm2m1;
    const Eigen::Vector3d m_dDeltaV;
    const Eigen::Vector3d m_dDeltaW;
    const double m_dWeight;
};

template <int _nPriorWidth>
struct PriorCostFunction
{
    PriorCostFunction(const Eigen::Matrix<double,_nPriorWidth,1> dPrior, const Eigen::Matrix<double,_nPriorWidth,_nPriorWidth> dInformation)
        : m_dPrior(dPrior),m_dInformation(dInformation)
    {}

    template <typename T>
    bool operator()(const T* const _tParams, T* _tResiduals) const
    {
        const Eigen::Map<const Eigen::Matrix<T,_nPriorWidth,1> > params(_tParams);
        Eigen::Map<Eigen::Matrix<T,_nPriorWidth,1> > residuals(_tResiduals);
        residuals = m_dInformation.template cast<T>()*(params-m_dPrior.template cast<T>());
        return true;
    }

    const Eigen::Matrix<double,_nPriorWidth,1> m_dPrior;
    const Eigen::Matrix<double,_nPriorWidth,_nPriorWidth> m_dInformation;
};

/// Calculates the IMU residual which is calculated using parameters from the PREVIOUS pose.
/// The parameter array should be as follows
/// [ 6d delta on the starting pose ] [ 3d velocity of starting pose ] [ 2d gravity vector ]

struct ImuCostFunction
{
    ImuCostFunction(const Sophus::SE3d& imuDelta, const Eigen::Vector3d& imuDeltaVel, const double& dt, const double dWeight)    :
        m_Tx1_x2(imuDelta),
        m_dDeltaV(imuDeltaVel),
        m_dT(dt),
        m_dWeight(dWeight)
  {
  }

    template <typename T>
    bool operator()(const T* const _tx2,const T* const _tx1,const T* const _tVx2,const T* const _tVx1,const T* const _tG, T* residuals) const
    {
        Eigen::IOFormat CleanFmt(3, 0, ", ", "\n" , "[" , "]");

        //the residual vector consists of a 6d pose and a 3d velocity residual
        Eigen::Map<Eigen::Matrix<T,6,1> > pose_residuals(residuals); //the pose residuals
        Eigen::Map<Eigen::Matrix<T,3,1> > vel_residuals(&residuals[6]); //the velocity residuals
        //Eigen::Map<Eigen::Matrix<T,3,1> > vel_prior(&residuals[9]); //the velocity residuals

        //parameter vector consists of a 6d pose delta plus starting velocity and 2d gravity angles
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_x2(_tx2);
        const Eigen::Map<const Sophus::SE3Group<T> > T_w_x1(_tx1);
        const Eigen::Map<const Sophus::SO3Group<T> > R_w_x1(&_tx1[0]);
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_v1(_tVx1); //the velocity at the starting point
        const Eigen::Map<const Eigen::Matrix<T,3,1> > v_v2(_tVx2); //the velocity at the end point
        const Eigen::Map<const Eigen::Matrix<T,2,1> > g(_tG); //the 2d gravity vector (angles)


        Sophus::SE3Group<T> imuT_w_x2 = T_w_x1*m_Tx1_x2.cast<T>();

        //get the gravity components in 3d based on the 2 angles of the gravity vector
        const Eigen::Matrix<T,3,1> g_vector = SensorFusionCeres::GetGravityVector<T>(g);
        //std::cout << "gravity vector is: " << g_vector.template cast<double>() << std::endl;

        //now augment the final pose based on the XYZ transformations of the acceleration and gravity influence
        imuT_w_x2.translation() += (-g_vector*(T)0.5*(T)m_dT*(T)m_dT + v_v1*(T)m_dT);

        //and now calculate the error with this pose
        pose_residuals = (imuT_w_x2 * T_w_x2.inverse()).log();
        pose_residuals.head(3) *= (T)g_dImuTranslationWeight;
        pose_residuals.tail(3) *= (T)g_dImuOrientationWeight;
        //if(m_dT > (T)0.1){
        //std::cout << "Residuals: [" << pose_residuals.transpose().format(CleanFmt) << "]" << " g: " << g_vector.transpose().format(CleanFmt) << " v1: " << v_v1.transpose().format(CleanFmt) << std::endl;
        //std::cout << "dt " << m_dT << " Residuals:" << pose_residuals.transpose().format(CleanFmt) << "" << " goalpose: " << T_w_x2.log().transpose().format(CleanFmt) << " imu: " << m_Tx1_x2.log().transpose().format(CleanFmt) << std::endl;
        //}

        //to calculate the velocity error, first augment the IMU integration velocity with gravity and initial velocity
        const Eigen::Matrix<T,3,1> v_end = v_v1 + (R_w_x1*m_dDeltaV.cast<T>()) + -g_vector*(T)m_dT;
        vel_residuals = (v_end - v_v2) * (T)g_dImuVelocityWeight;//  * (T)(100/m_dT);
        return true;
    }


    const Sophus::SE3d m_Tx1_x2;
    const Eigen::Vector3d m_dDeltaV;
    const double m_dT;
    const double m_dWeight;
};
}

#endif // COSTFUNCTIONS_H
