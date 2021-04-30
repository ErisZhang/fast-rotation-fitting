#include <Eigen/Core>
#include <iostream>

template <typename T>
void mbcm(
    const Eigen::Matrix<T,3,3> &A,
    const unsigned int maxIter,
    Eigen::Matrix<T,3,3> &R) {

    Eigen::Quaternion<T> q(R); // warm start

    for (int iter = 0; iter < maxIter; iter++) {
        Eigen::Matrix<T,3,3> R = q.matrix();
        Eigen::Matrix<T,3,1> omega = (R.col(0).cross(A.col(0)) + R.col(1).cross(A.col(1)) + R.col(2).cross(A.col(2)) ) * (1.0 / fabs(R.col(0).dot(A.col(0)) + R.col (1).dot(A.col(1)) + R.col(2).dot(A.col(2))) + 1.0e-9);
        T w = omega.norm();
        if (w < 1.0e-9) {
            break;
        }

        q = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(w, (1.0 / w) * omega)) * q;
        q.normalize();
    }

    R = q.matrix();
}