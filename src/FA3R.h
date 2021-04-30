#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <stdint.h>


const uint8_t shift = 20;
const int64_t shifted1 = ((int64_t) 2) << shift;
const int64_t shifted2 = shift << 1;
const int64_t shifted3 = ((int64_t) 2) << (((int64_t) 3) * shift + 1);


void cross(const int64_t &x1, const int64_t &x2, const int64_t &x3, 
           const int64_t &y1, const int64_t &y2, const int64_t &y3, 
           const int64_t &k, int64_t *z1, int64_t *z2, int64_t *z3)
{
    *z1 = (k * (*z1 + ((x2 * y3 - x3 * y2) >> shift))) >> shifted2;
    *z2 = (k * (*z2 + ((x3 * y1 - x1 * y3) >> shift))) >> shifted2;
    *z3 = (k * (*z3 + ((x1 * y2 - x2 * y1) >> shift))) >> shifted2;
}


template <typename T>
void cross(const Eigen::Matrix<T,3,1> &x, const Eigen::Matrix<T,3,1> &y, const T &k, Eigen::Matrix<T,3,1> &z)
{
    z(0) = k * (z(0) + x(1) * y(2) - x(2) * y(1));
    z(1) = k * (z(1) + x(2) * y(0) - x(0) * y(2));
    z(2) = k * (z(2) + x(0) * y(1) - x(1) * y(0));
}


template <typename T>
void FA3R(Eigen::Matrix<T,3,3> * sigma,
          		   int num,
		   Eigen::Matrix<T,3,3> * rRes,
		   Eigen::Matrix<T,3,1> * tRes)
{
    Eigen::Matrix<T,3,3> * sigma_ = sigma;
    Eigen::Matrix<T,3,1> mean_X, mean_Y;
    
    Eigen::Matrix<T,3,1> hx((*sigma_)(0, 0), (*sigma_)(1, 0), (*sigma_)(2, 0));
    Eigen::Matrix<T,3,1> hy((*sigma_)(0, 1), (*sigma_)(1, 1), (*sigma_)(2, 1));
    Eigen::Matrix<T,3,1> hz((*sigma_)(0, 2), (*sigma_)(1, 2), (*sigma_)(2, 2));
    Eigen::Matrix<T,3,1> hx_, hy_, hz_;
    T k;
    
    for(int i = 0; i < num; ++i)
    {
        k = 2.0 / (hx(0) * hx(0) + hx(1) * hx(1) + hx(2) * hx(2) +
                   hy(0) * hy(0) + hy(1) * hy(1) + hy(2) * hy(2) +
                   hz(0) * hz(0) + hz(1) * hz(1) + hz(2) * hz(2) + 1.0);
        
        hx_ = hx;  hy_ = hy; hz_ = hz;
        
        cross(hx_, hy_, k, hz);
        cross(hz_, hx_, k, hy);
        cross(hy_, hz_, k, hx);
    }

    (*rRes)(0, 0) = hx(0);  (*rRes)(0, 1) = hy(0);  (*rRes)(0, 2) = hz(0);
    (*rRes)(1, 0) = hx(1);  (*rRes)(1, 1) = hy(1);  (*rRes)(1, 2) = hz(1);
    (*rRes)(2, 0) = hx(2);  (*rRes)(2, 1) = hy(2);  (*rRes)(2, 2) = hz(2);
	*tRes = mean_X - (*rRes).transpose() * mean_Y;

}