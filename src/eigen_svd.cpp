#include <Eigen/Dense>
#include <Eigen/Core>

int eigen_svd(const Eigen::Matrix3f & M, Eigen::Matrix3f & R)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV );
    const Eigen::Matrix3f U = svd.matrixU();
    const Eigen::Matrix3f V = svd.matrixV();
    R = U*V.transpose();
    // Check for reflection
    if(R.determinant() < 0)
    {
      // flip last singular vector
      static const Eigen::Matrix3f F = 
        (Eigen::Matrix3f()<<1,0,0,0,1,0,0,0,-1).finished();
      R = U*F*V.transpose();
    }
    return 0;
}
