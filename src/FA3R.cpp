#include "FA3R.h"
#include <stdint.h>
#include <time.h>  

const uint8_t shift = 20;
const float d2l = pow(2, (float) shift);
const int64_t shifted1 = ((int64_t) 2) << shift;
const int64_t shifted2 = shift << 1;
const int64_t shifted3 = ((int64_t) 2) << (((int64_t) 3) * shift + 1);
const float l2d = 1.0 / d2l;


void cross(const int64_t &x1, const int64_t &x2, const int64_t &x3, 
           const int64_t &y1, const int64_t &y2, const int64_t &y3, 
           const int64_t &k, int64_t *z1, int64_t *z2, int64_t *z3)
{
    *z1 = (k * (*z1 + ((x2 * y3 - x3 * y2) >> shift))) >> shifted2;
    *z2 = (k * (*z2 + ((x3 * y1 - x1 * y3) >> shift))) >> shifted2;
    *z3 = (k * (*z3 + ((x1 * y2 - x2 * y1) >> shift))) >> shifted2;
}

void cross(const Vector3f &x, const Vector3f &y, const float &k, Vector3f &z)
{
    z(0) = k * (z(0) + x(1) * y(2) - x(2) * y(1));
    z(1) = k * (z(1) + x(2) * y(0) - x(0) * y(2));
    z(2) = k * (z(2) + x(0) * y(1) - x(1) * y(0));
}


void FA3R(Matrix3f * sigma,
                   int num,
		   Matrix3f * rRes,
		   Vector3f * tRes)
{
    Matrix3f * sigma_ = sigma;
    Vector3f mean_X, mean_Y;
    
    Vector3f hx((*sigma_)(0, 0), (*sigma_)(1, 0), (*sigma_)(2, 0));
    Vector3f hy((*sigma_)(0, 1), (*sigma_)(1, 1), (*sigma_)(2, 1));
    Vector3f hz((*sigma_)(0, 2), (*sigma_)(1, 2), (*sigma_)(2, 2));
    Vector3f hx_, hy_, hz_;
    float k;
    
    
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
