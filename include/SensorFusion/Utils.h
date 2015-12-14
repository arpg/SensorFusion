#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>

namespace Eigen
{
    typedef Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXdRowMaj;
    typedef std::vector<Eigen::MatrixXdRowMaj,Eigen::aligned_allocator<Eigen::MatrixXdRowMaj> > MatrixXdRowMajAlignedVec;
    typedef std::vector<Eigen::Vector6d,Eigen::aligned_allocator<Eigen::Vector6d> > Vector6dAlignedVec;
    typedef std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dAlignedVec;
}

namespace fusion
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    inline T powi(T num, int exp) {
        if(exp == 0 ){
            return 1;
        }else if( exp < 0 ) {
            return 0;
        }else{
            T ret = num;
            for(int ii = 1; ii < exp ; ii++) {
                ret *= num;
            }
            return ret;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    inline double powi(double num, int exp) { return powi<double>(num,exp); }


    ///////////////////////////////////////////////////////////////////////
    inline double AngleWrap( double d )
    {
        while( d > M_PI ) {
            d -= 2*M_PI;
        }
        while( d < -M_PI ) {
            d += 2*M_PI;
        }
        return d;
    }
}

#endif // UTILS_H
