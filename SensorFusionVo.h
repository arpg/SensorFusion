#ifndef SENSORFUSIONVO_H
#define SENSORFUSIONVO_H

#include <stdio.h>
#include "Eigen/Eigen"
#include "Eigen/Sparse"
#include "boost/thread.hpp"
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include <SE3.h>
#include "LocalParamSe3.h"
#include "Utils.h"

#include "float.h"

struct ProjectionCostFunction
{
    ProjectionCostFunction(const Eigen::Vector2d& dMeasurement, const Sophus::SE3d& dTsi, const Eigen::Matrix3d& dK, const double dWeight  )
        : m_dMeas_uv(dMeasurement),
          m_dWeight(dWeight),
          m_dK(dK),
          m_dTsi(dTsi)
    {

    }

    template <typename T>
    bool operator()(const T* const _tTiw,const T* const _tXp, const T* const _tTpw, T* residuals) const
    {
        //project the position of the landmark into the current coordinate system
        const Eigen::Map<const Sophus::SE3Group<T> > Tpw(_tTpw);    //transform from datum to the pose where the landmark was added
        const Eigen::Map<const Sophus::SE3Group<T> > Tiw(_tTiw);    //transform from datum to the pose where the measurement was taken
        const Eigen::Map<Eigen::Matrix<T,3,1> > Xp(_tXp);   //offset from the pose where landmark was added to the pose

        Eigen::Map<Eigen::Matrix<T,2,1> > uv_residuals(residuals); //the image space residuals
        Eigen::Matrix<T,4,1> Xp_homog;
        //put into homogenous coordinates
        Xp_homog.head(3) = Xp;
        Xp_homog[3] = 1;
        Eigen::Matrix<T,4,1> Xw = Tpw.inverse() * Xp_homog; //offset from datum to landmark

        //first get the sensor coordinates in world coordinates transform
        Sophus::SE3Group<T> Tsw = m_dTsi.cast<T>() * Tiw;
        //get the landmark in the sensor coordaintes
        Eigen::Matrix<T,4,1> Xs = Tsw * Xw;
        //transform to computer vision frame
        Eigen::Matrix<T,3,1> Xs_cv(Xs(1),Xs(2),Xs(0));

        //now do the transformation into image space
        Eigen::Matrix<T,3,1> proj_uv = m_dK * Xs_cv;
        proj_uv /= proj_uv[2];

        uv_residuals = proj_uv.head(2) - m_dMeas_uv;
        return true;
    }

    const Eigen::Vector2d m_dMeas_uv;
    const double m_dWeight;
    const Eigen::Matrix3d m_dK;    
    const Sophus::SE3d m_dTsi;
};

class SensorFusionVo
{
public:
    SensorFusionVo();
};

#endif // SENSORFUSIONVO_H
