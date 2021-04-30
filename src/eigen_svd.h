#include <Eigen/Dense>

template <typename T>
void eigen_svd(
  const Eigen::Matrix<T,3,3> & M,
    Eigen::Matrix<T,3,3> & R)
{
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV );
    const Eigen::Matrix<T,3,3> U = svd.matrixU();
    const Eigen::Matrix<T,3,3> V = svd.matrixV();
    R = U*V.transpose();
    // Check for reflection
    if(R.determinant() < 0)
    {
      // flip last singular vector
      static const Eigen::Matrix<T,3,3> F = 
        (Eigen::Matrix<T,3,3>()<<1,0,0,0,1,0,0,0,-1).finished();
      R = U*F*V.transpose();
    }
}