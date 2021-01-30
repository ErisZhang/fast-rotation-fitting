#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// Solution to orthogonal procrustes problem using unit quaternions according to
// "Closed-form solution of absolute orientation using Unit Quaternions" [Horn
// 1986]
//
// Inputs:
//   M  3x3 covariance matrix
// Outputs:
//   R  3x3 nearest rotation matrix
//
template <typename T>
inline void horneig(
  const Eigen::Matrix<T,3,3> & S,
  Eigen::Matrix<T,3,3> & R)
{
  // Adapted from leastSquareErrorRigidTransformat by The Mobile Robot
  // Programming Toolkit (MRPT) C++ library 

  // Only the lower part is referenced by SelfAdjointEigenSolver
  Eigen::Matrix<T,4,4> N;
  N(0,0) = S(0,0) + S(1,1) + S(2,2);
  N(1,0) = S(1,2) - S(2,1);
  N(2,0) = S(2,0) - S(0,2);
  N(3,0) = S(0,1) - S(1,0);
  //N(0,1) = N(0,1);
  N(1,1) = S(0,0) - S(1,1) - S(2,2);
  N(2,1) = S(0,1) + S(1,0);
  N(3,1) = S(2,0) + S(0,2);
  //N(0,2) = N(0,2);
  //N(1,2) = N(1,2);
  N(2,2) = -S(0,0) + S(1,1) - S(2,2);
  N(3,2) = S(1,2) + S(2,1);
  //N(0,3) = N(0,3);
  //N(1,3) = N(1,3);
  //N(2,3) = N(2,3);
  N(3,3) = -S(0,0) - S(1,1) + S(2,2);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,4,4> > eig(N);
  const Eigen::Matrix<T,4,1> Z = eig.eigenvectors().col(3);
  const T yy2 = 2.0 * Z(2) * Z(2);
  const T xy2 = 2.0 * Z(3) * Z(2);
  const T xz2 = 2.0 * Z(3) * Z(1);
  const T yz2 = 2.0 * Z(2) * Z(1);
  const T zz2 = 2.0 * Z(1) * Z(1);
  const T wz2 = 2.0 * Z(0) * Z(1);
  const T wy2 = 2.0 * Z(0) * Z(2);
  const T wx2 = 2.0 * Z(0) * Z(3);
  const T xx2 = 2.0 * Z(3) * Z(3);
  R(2,2) = - yy2 - zz2 + 1.0;
  R(1,2) = xy2 + wz2;
  R(0,2) = xz2 - wy2;
  R(2,1) = xy2 - wz2;
  R(1,1) = - xx2 - zz2 + 1.0;
  R(0,1) = yz2 + wx2;
  R(2,0) = xz2 + wy2;
  R(1,0) = yz2 - wx2;
  R(0,0) = - xx2 - yy2 + 1.0;
}
