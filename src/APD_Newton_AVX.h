#include "AVX_math.h"

using namespace std;
using namespace Eigen;

// ----------------------------------------------------------------------------------------------
//computes the APD of 8 deformation gradients. (Alg. 3 from the paper)
void APD_Newton_AVX(const Vector3f8& F1, const Vector3f8& F2, const Vector3f8& F3, Quaternion8f& q)
{
	//one iteration is sufficient for plausible results
	for (int it = 0; it < 1; it++)
	{
		//transform quaternion to rotation matrix
		Matrix3f8 R;
		q.toRotationMatrix(R);

		//columns of B = RT * F
		Vector3f8 B0 = R.transpose() * F1;
		Vector3f8 B1 = R.transpose() * F2;
		Vector3f8 B2 = R.transpose() * F3;

		Vector3f8 gradient(B2[1] - B1[2], B0[2] - B2[0], B1[0] - B0[1]);

		//compute Hessian, use the fact that it is symmetric
		Scalarf8 h00 = B1[1] + B2[2];
		Scalarf8 h11 = B0[0] + B2[2];
		Scalarf8 h22 = B0[0] + B1[1];
		Scalarf8 h01 = Scalarf8(-0.5) * (B1[0] + B0[1]);
		Scalarf8 h02 = Scalarf8(-0.5) * (B2[0] + B0[2]);
		Scalarf8 h12 = Scalarf8(-0.5) * (B2[1] + B1[2]);

		Scalarf8 detH = Scalarf8(-1.0) * h02 * h02 * h11 + Scalarf8(2.0) * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22;

		Vector3f8 omega;
		//compute symmetric inverse
		const Scalarf8 factor = Scalarf8(-0.25) / detH;
		omega[0] = (h11 * h22 - h12 * h12) * gradient[0]
			+ (h02 * h12 - h01 * h22) * gradient[1]
			+ (h01 * h12 - h02 * h11) * gradient[2];
		omega[0] *= factor;

		omega[1] = (h02 * h12 - h01 * h22) * gradient[0]
			+ (h00 * h22 - h02 * h02) * gradient[1]
			+ (h01 * h02 - h00 * h12) * gradient[2];
		omega[1] *= factor;

		omega[2] = (h01 * h12 - h02 * h11) * gradient[0]
			+ (h01 * h02 - h00 * h12) * gradient[1]
			+ (h00 * h11 - h01 * h01) * gradient[2];
		omega[2] *= factor;

        // potentially buggy
		omega = Vector3f8::blend(abs(detH) < 1.0e-9f, gradient * Scalarf8(-1.0), omega);	//if det(H) = 0 use gradient descent, never happened in our tests, could also be removed 

		//instead of clamping just use gradient descent. also works fine and does not require the norm
		Scalarf8 useGD = blend(omega * gradient > Scalarf8(0.0), Scalarf8(1.0), Scalarf8(-1.0));
		omega = Vector3f8::blend(useGD > Scalarf8(0.0), gradient * Scalarf8(-0.125), omega);

		Scalarf8 l_omega2 = omega.lengthSquared();
		const Scalarf8 w = (1.0 - l_omega2) / (1.0 + l_omega2);
		const Vector3f8 vec = omega * (2.0 / (1.0 + l_omega2));
		q = q * Quaternion8f(vec.x(), vec.y(), vec.z(), w);		//no normalization needed because the Cayley map returs a unit quaternion
	}
    
}