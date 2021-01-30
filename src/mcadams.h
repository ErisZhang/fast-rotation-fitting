#include <Eigen/Core>

#undef USE_SCALAR_IMPLEMENTATION
#undef USE_SSE_IMPLEMENTATION
#define USE_AVX_IMPLEMENTATION
#define COMPUTE_U_AS_MATRIX
#define COMPUTE_V_AS_MATRIX
#include "Singular_Value_Decomposition_Preamble.hpp"

// Thin wrapper on Sifakis code. Just does SVD. Does not compute R.
void mcadams_svd_avx(
  float * Ap,
  float * Up,
  float * Sp,
  float * Vp);
// Computes R = U*V'
void mcadams_avx(
  float * Ap,
  float * Rp);

#pragma runtime_checks( "u", off )  // disable runtime asserts on xor eax,eax type of stuff (doesn't always work, disable explicitly in compiler settings)
void mcadams_svd_avx(
  float * Ap,
  float * Up,
  float * Sp,
  float * Vp)
{
#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

  ENABLE_AVX_IMPLEMENTATION(Va11=_mm256_loadu_ps(Ap+8*0);)
  ENABLE_AVX_IMPLEMENTATION(Va21=_mm256_loadu_ps(Ap+8*1);)
  ENABLE_AVX_IMPLEMENTATION(Va31=_mm256_loadu_ps(Ap+8*2);)
  ENABLE_AVX_IMPLEMENTATION(Va12=_mm256_loadu_ps(Ap+8*3);)
  ENABLE_AVX_IMPLEMENTATION(Va22=_mm256_loadu_ps(Ap+8*4);)
  ENABLE_AVX_IMPLEMENTATION(Va32=_mm256_loadu_ps(Ap+8*5);)
  ENABLE_AVX_IMPLEMENTATION(Va13=_mm256_loadu_ps(Ap+8*6);)
  ENABLE_AVX_IMPLEMENTATION(Va23=_mm256_loadu_ps(Ap+8*7);)
  ENABLE_AVX_IMPLEMENTATION(Va33=_mm256_loadu_ps(Ap+8*8);)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"

  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*0,Vu11);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*1,Vu21);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*2,Vu31);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*3,Vu12);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*4,Vu22);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*5,Vu32);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*6,Vu13);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*7,Vu23);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*8,Vu33);)

  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*0,Vv11);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*1,Vv21);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*2,Vv31);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*3,Vv12);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*4,Vv22);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*5,Vv32);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*6,Vv13);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*7,Vv23);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*8,Vv33);)

  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Sp+8*0,Va11);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Sp+8*1,Va22);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Sp+8*2,Va33);)

}
#pragma runtime_checks( "u", restore )

#pragma runtime_checks( "u", off )  // disable runtime asserts on xor eax,eax type of stuff (doesn't always work, disable explicitly in compiler settings)
void mcadams_avx(
  float * Ap,
  float * Rp)
{
  // this code assumes USE_AVX_IMPLEMENTATION is defined
  //float Ashuffle[9][8];
  //for (int i=0; i<3; i++)
  //{
  //  for (int j=0; j<3; j++)
  //  {
  //    for (int k=0; k<8; k++)
  //    {
  //      Ashuffle[i + j*3][k] = A(i + 3*k, j);
  //    }
  //  }
  //}
  //float Ushuffle[9][8], Vshuffle[9][8], Sshuffle[3][8];

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"

  ENABLE_AVX_IMPLEMENTATION(Va11=_mm256_loadu_ps(Ap+8*0);)
  ENABLE_AVX_IMPLEMENTATION(Va21=_mm256_loadu_ps(Ap+8*1);)
  ENABLE_AVX_IMPLEMENTATION(Va31=_mm256_loadu_ps(Ap+8*2);)
  ENABLE_AVX_IMPLEMENTATION(Va12=_mm256_loadu_ps(Ap+8*3);)
  ENABLE_AVX_IMPLEMENTATION(Va22=_mm256_loadu_ps(Ap+8*4);)
  ENABLE_AVX_IMPLEMENTATION(Va32=_mm256_loadu_ps(Ap+8*5);)
  ENABLE_AVX_IMPLEMENTATION(Va13=_mm256_loadu_ps(Ap+8*6);)
  ENABLE_AVX_IMPLEMENTATION(Va23=_mm256_loadu_ps(Ap+8*7);)
  ENABLE_AVX_IMPLEMENTATION(Va33=_mm256_loadu_ps(Ap+8*8);)

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"

  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*0,Vu11);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*1,Vu21);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*2,Vu31);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*3,Vu12);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*4,Vu22);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*5,Vu32);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*6,Vu13);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*7,Vu23);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Up+8*8,Vu33);)

  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*0,Vv11);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*1,Vv21);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*2,Vv31);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*3,Vv12);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*4,Vv22);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*5,Vv32);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*6,Vv13);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*7,Vv23);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Vp+8*8,Vv33);)

  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Sp+8*0,Va11);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Sp+8*1,Va22);)
  //ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Sp+8*2,Va33);)
  

  // // save R
  //r11_ = r11;
  //r12_ = r12;
  //r13_ = r13;
  //r21_ = r21;
  //r22_ = r22;
  //r23_ = r23;
  //r31_ = r31;
  //r32_ = r32;
  //r33_ = r33;
  

  // update R
  __m256 r11 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu11, Vv11), _mm256_mul_ps(Vu12, Vv12)), _mm256_mul_ps(Vu13, Vv13));
  __m256 r12 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu11, Vv21), _mm256_mul_ps(Vu12, Vv22)), _mm256_mul_ps(Vu13, Vv23));
  __m256 r13 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu11, Vv31), _mm256_mul_ps(Vu12, Vv32)), _mm256_mul_ps(Vu13, Vv33));
  __m256 r21 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu21, Vv11), _mm256_mul_ps(Vu22, Vv12)), _mm256_mul_ps(Vu23, Vv13));
  __m256 r22 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu21, Vv21), _mm256_mul_ps(Vu22, Vv22)), _mm256_mul_ps(Vu23, Vv23));
  __m256 r23 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu21, Vv31), _mm256_mul_ps(Vu22, Vv32)), _mm256_mul_ps(Vu23, Vv33));
  __m256 r31 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu31, Vv11), _mm256_mul_ps(Vu32, Vv12)), _mm256_mul_ps(Vu33, Vv13));
  __m256 r32 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu31, Vv21), _mm256_mul_ps(Vu32, Vv22)), _mm256_mul_ps(Vu33, Vv23));
  __m256 r33 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Vu31, Vv31), _mm256_mul_ps(Vu32, Vv32)), _mm256_mul_ps(Vu33, Vv33));
 
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*0,r11);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*1,r21);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*2,r31);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*3,r12);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*4,r22);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*5,r32);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*6,r13);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*7,r23);)
  ENABLE_AVX_IMPLEMENTATION(_mm256_storeu_ps(Rp+8*8,r33);)

  //for (int i=0; i<3; i++)
  //{
  //  for (int j=0; j<3; j++)
  //  {
  //    for (int k=0; k<8; k++)
  //    {
  //      U(i + 3*k, j) = Ushuffle[i + j*3][k];
  //      V(i + 3*k, j) = Vshuffle[i + j*3][k];
  //    }
  //  }
  //}

  //for (int i=0; i<3; i++)
  //{
  //  for (int k=0; k<8; k++)
  //  {
  //    S(i + 3*k, 0) = Sshuffle[i][k];
  //  }
  //}
}
#pragma runtime_checks( "u", restore )

