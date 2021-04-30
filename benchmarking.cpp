#include <stdio.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <stdlib.h>
#include <fstream>
#include <float.h>

#include "eigen_svd.h"
#include "FA3R.h"
#include "horneig.h"
#include "mcadams.h"
#include "MBCM.h"
#include "fit_rotation_avx.h"
#include "AVX_math.h"
#include "APD_Newton_AVX.h"

#include <igl/get_seconds.h>
#include <igl/readDMAT.h>
#include <igl/polar_svd3x3.h>


#ifndef __AVX2__
#error "AVX2 not found. Re-build this project with -mavx2"
#endif


using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) 
{
  double 
    t_eigen_svd,
    t_horn_eig,
    t_fa3r,
    t_mbcm,
    t_polar,
    t_mcadams_svd_avx,
    t_mcadams_avx,
    t_apd_newton_avx,
    t_cayley,
    t_cayley_avx,
    t_cayley_warm_start,
    t_cayley_warm_start_avx;

    using nano = std::chrono::nanoseconds;
    MatrixXd M;

    std::string filename = "../data/M_decimated_knight.dmat";
    // std::string filename = "../data/M.dmat";
    if(!igl::readDMAT(filename,M))
    {
      std::cerr<<"Failed to read "<<filename<<std::endl;
      return EXIT_FAILURE;
    }
    if(M.cols() != 3 || (M.rows()%3) != 0)
    {
      std::cerr<<"M must be n*3 by 3"<<std::endl;
      return EXIT_FAILURE;
    }
    printf("Read %d*3 by %d matrix from %s.\n",
      (int)M.rows()/3,(int)M.cols(),filename.c_str());


    if(M.rows() % 3*8 != 0)
    {
      M.conservativeResize((M.rows()/(3*8))*(3*8),3);
      printf("Truncating to %d*3 by %d matrix.\n",
        (int)M.rows()/3,(int)M.cols());
    }
    int num_repeat = 50;
    M = M.replicate(num_repeat,1).eval();


    int nM = M.rows()/3;
    int num_of_group = nM/8;
    int num_of_ele = num_of_group*8*9;
    // aligned malloc memory for M
    void *Ms = NULL;
    // Copy and cast into single precision
    Eigen::MatrixXf Mfloat = M.cast<float>();
    // aligned malloc memory for R
    // don't forget to initialize R::Identity !!!!!!
    void *R = NULL;
    // Allocate memory for Sifakis outputs
    void *U = NULL;
    void *V = NULL;
    void *S = NULL;
#if defined(_MSC_VER)
    Ms = _aligned_malloc(num_of_ele * sizeof(float), 32);
    R = _aligned_malloc(num_of_ele * sizeof(float), 32);
    U = _aligned_malloc(num_of_ele * sizeof(float), 32);
    V = _aligned_malloc(num_of_ele * sizeof(float), 32);
    S = _aligned_malloc(num_of_group*8*3* sizeof(float), 32);
#else
    posix_memalign(&Ms, 32, num_of_ele * sizeof(float));
    posix_memalign(&R, 32, num_of_ele * sizeof(float));
    posix_memalign(&U, 32, num_of_ele * sizeof(float));
    posix_memalign(&V, 32, num_of_ele * sizeof(float));
    posix_memalign(&S, 32, num_of_group*8*3* sizeof(float));
#endif
    float *Mf = (float *)Ms;
    float *Rf = (float *)R;
    float *Uf = (float *)U;
    float *Vf = (float *)V;
    float *Sf = (float *)S;
    std::vector<Quaternion8f, AlignmentAllocator<Quaternion8f, 32>> quats;
    // std::vector<Vector3f8> F1s, F2s, F3s;


    const auto init = [&]()
    {
      // Copy and cast into memory shuffled M
      for (int i = 0; i < num_of_group; i++) {
          for (int j = 0; j < 9; j++) {
              Mf[i*72+j*8+0] = (float)M(i*24+0*3+j/3, j%3);
              Mf[i*72+j*8+1] = (float)M(i*24+1*3+j/3, j%3);
              Mf[i*72+j*8+2] = (float)M(i*24+2*3+j/3, j%3);
              Mf[i*72+j*8+3] = (float)M(i*24+3*3+j/3, j%3);
              Mf[i*72+j*8+4] = (float)M(i*24+4*3+j/3, j%3);
              Mf[i*72+j*8+5] = (float)M(i*24+5*3+j/3, j%3);
              Mf[i*72+j*8+6] = (float)M(i*24+6*3+j/3, j%3);
              Mf[i*72+j*8+7] = (float)M(i*24+7*3+j/3, j%3);
          }
      }
      for (int i = 0; i < num_of_group; i++) {
          for (int j = 0; j < 8; j++) {
              Rf[i*72+0*8+j] = 1.0f;
              Rf[i*72+1*8+j] = 0.0f;
              Rf[i*72+2*8+j] = 0.0f;
              Rf[i*72+3*8+j] = 0.0f;
              Rf[i*72+4*8+j] = 1.0f;
              Rf[i*72+5*8+j] = 0.0f;
              Rf[i*72+6*8+j] = 0.0f;
              Rf[i*72+7*8+j] = 0.0f;
              Rf[i*72+8*8+j] = 1.0f;
          }
      }
    };


    const auto initQuats = [&]()
    {
      // init quats
      Quaternionr Rint = Quaternionr::Identity();
      quats.resize(num_of_group);
      for (int i = 0; i < num_of_group; i++) 
        quats[i] = Quaternion8f((float)Rint.x(), (float)Rint.y(), 
                    (float)Rint.z(), (float)Rint.w());
    };


    printf("\nRaw Times\n");
    printf("-----------------------------------------\n");

    // Eigen SVD
    {
      Matrix3f R_svd;
      for(int pass = 0;pass<2;pass++)
      {
        const double tic = igl::get_seconds();
        for (int i = 0; i < nM ; i++) 
        {
          const Eigen::Matrix3f Mi = Mfloat.block(i*3, 0, 3, 3);
          eigen_svd(Mi, R_svd);
        }
        if(pass==1)
        {
          t_eigen_svd = (igl::get_seconds()-tic);
        }
      }
    }
    printf("Eigen SVD                %5f seconds\n",t_eigen_svd);

    // Horn eig
    {
      Matrix3f R_svd;
      for(int pass = 0;pass<2;pass++)
      {
        const double tic = igl::get_seconds();
        for (int i = 0; i < nM ; i++) 
        {
          const Eigen::Matrix3f Mi = Mfloat.block(i*3, 0, 3, 3);
          horneig(Mi, R_svd);
        }
        if(pass==1)
        {
          t_horn_eig = (igl::get_seconds()-tic);
        }
      }
    }
    printf("Horn eig                 %5f seconds\n",t_horn_eig);

    // FA3R
    {
      Matrix3f R_svd;
      Vector3f tRes;
      for(int pass = 0;pass<2;pass++)
      {
        const double tic = igl::get_seconds();
        for (int i = 0; i < nM ; i++) 
        {
          Eigen::Matrix3f Mi = Mfloat.block(i*3, 0, 3, 3);
          FA3R(&Mi, 15, &R_svd, &tRes);
        }
        if(pass==1)
        {
          t_fa3r = (igl::get_seconds()-tic);
        }
      }
    }
    printf("FA3R                     %5f seconds\n",t_fa3r);


    // MBCM
    {
      for(int pass = 0;pass<2;pass++)
      {
        Matrix3f R_polar;
        R_polar = Eigen::MatrixXf::Identity(3,3);
        const double tic = igl::get_seconds();
        for (int i = 0; i < nM ; i++) 
        {
          Eigen::Matrix3f Mi = Mfloat.block(i*3, 0, 3, 3);
          mbcm(Mi, 5, R_polar);
        }
        if(pass==1)
        {
          t_mbcm = (igl::get_seconds()-tic);
        }
      }
    }
    printf("MBCM                     %5f seconds\n",t_mbcm);


    // polar decomposition AVX
    {
      for(int pass = 0; pass < 2; pass++)
      {
        Matrix3f R_polar;
        const double tic = igl::get_seconds();
        for (int i = 0; i < nM ; i++) 
        {
          Eigen::Matrix3f Mi = Mfloat.block(i*3, 0, 3, 3);
          igl::polar_svd3x3(Mi, R_polar);
        }
        if(pass==1)
        {
          t_polar = (igl::get_seconds()-tic);
        }
      }
    }
    printf("Polar decomp +avx        %5f seconds\n",t_polar);

    // Sifakis AVX
    {
      Matrix<float,3*8,3> M_8(3*8, 3);
      Matrix<float,3*8,3> R_8(3*8, 3);
      for(int pass = 0;pass<2;pass++)
      {
        init();
        //Mfloat = M.cast<float>();
        const double tic = igl::get_seconds();
        for (int i = 0; i < num_of_group; i++) {
            mcadams_svd_avx(Mf+i*72, Uf+i*72, Sf+i*24, Vf+i*72);
        }
        if(pass==1)
        {
          t_mcadams_svd_avx = (igl::get_seconds()-tic);
        }
      }
    }
    printf("McAdams SVD +avx         %5f seconds\n",t_mcadams_svd_avx);
 
    // Sifakis AVX
    {
      Matrix<float,3*8,3> M_8(3*8, 3);
      Matrix<float,3*8,3> R_8(3*8, 3);
      for(int pass = 0;pass<2;pass++)
      {
        init();
        //Mfloat = M.cast<float>();
        const double tic = igl::get_seconds();
        for (int i = 0; i < num_of_group; i++) {
            //M_8 = Mfloat.block(i*24, 0, 24, 3);
            mcadams_avx(Mf+i*72, R_8.data());
        }
        if(pass==1)
        {
          t_mcadams_avx = (igl::get_seconds()-tic);
        }
      }
    }
    printf("McAdams R +avx           %5f seconds\n",t_mcadams_avx);
    
    
    // Cayley
    for(int pass = 0;pass<2;pass++)
    {
      init();
      const double tic = igl::get_seconds();
      fit_rotation_no_avx(Mf, Rf, num_of_group);
      if(pass==1)
      {
        t_cayley = (igl::get_seconds()-tic);
      }
    }
    printf("Cayley Gershgorin             %5f seconds\n",t_cayley);

    // Cayley +avx
    for(int pass = 0;pass<2;pass++)
    {
      init();
      Matrix<float,3*8,3> R_8(3*8, 3);
      const double tic = igl::get_seconds();
      fit_rotation_avx(Mf, Rf, num_of_group);
      if(pass==1)
      {
        t_cayley_avx = (igl::get_seconds()-tic);
      }
    }
    printf("Cayley Gershgorin + avx              %5f seconds\n",t_cayley_avx);

    // Cayley + warm-start 
    for(int pass = 0;pass<2;pass++)
    {
      init();
      const double tic = igl::get_seconds();
      fit_rotation_small_no_avx(Mf, Rf, num_of_group);
      if(pass==1)
      {
        t_cayley_warm_start = (igl::get_seconds()-tic);
      }
    }
    printf("Cayley Conservative + warm-start       %5f seconds\n",t_cayley_warm_start);

    // Cayley +warm-start +avx
    for(int pass = 0;pass<2;pass++)
    {
      init();
      const double tic = igl::get_seconds();
      fit_rotation_small_avx(Mf, Rf, num_of_group);
      if(pass==1)
      {
        t_cayley_warm_start_avx = (igl::get_seconds()-tic);
      }
    }
    printf("Cayley Conservative +warm-start + avx %5f seconds\n",t_cayley_warm_start_avx);


    // KBB 18: https://github.com/InteractiveComputerGraphics/FastCorotatedFEM
    {
      for (int pass = 0;pass<2;pass++)
      {
        init();
        initQuats();
        //transform quaternion to rotation matrix
        Vector3f8 R1, R2, R3;	//columns of the rotation matrix

        const double tic = igl::get_seconds();

        for (int i = 0; i < num_of_group; i++) {
            Quaternion8f & q = quats[i];
            // Vector3f8 F1, F2, F3;
            // F1 = F1s[i];
            // F2 = F2s[i];
            // F3 = F3s[i];
            Vector3f8 F1, F2, F3;
            for (int j = 0; j < 9; j++) {
                Scalarf8 s(Mf[i*72+j*8+0], Mf[i*72+j*8+1], Mf[i*72+j*8+2], Mf[i*72+j*8+3], Mf[i*72+j*8+4], Mf[i*72+j*8+5], Mf[i*72+j*8+6], Mf[i*72+j*8+7]);
                if (j / 3 == 0) {
                  F1[j % 3] = s;
                }
                else if (j / 3 == 1) {
                  F2[j % 3] = s;
                }
                else {
                  F3[j % 3] = s;
                }
            }

            APD_Newton_AVX(F1, F2, F3, q);

            // KBB18 code has this step too
            quats[i].toRotationMatrix(R1, R2, R3);

        }
        if (pass==1)
        {
          t_apd_newton_avx = (igl::get_seconds()-tic);
        }
      }
    }
    printf("Cayley (APD) + warm-start + avx       %5f seconds\n",t_apd_newton_avx);


    printf("\nSpeedup over Eigen SVD\n");
    printf("-------------------------------\n");
    printf("Eigen SVD                                       %5.1fx\n",t_eigen_svd/t_eigen_svd);
    printf("Horn eig                                        %5.1fx\n",t_eigen_svd/t_horn_eig);
    printf("FA3R                                            %5.1fx\n",t_eigen_svd/t_fa3r);
    printf("MBCM                                            %5.1fx\n",t_eigen_svd/t_mbcm);
    printf("Polar Decomp +avx                               %5.1fx\n",t_eigen_svd/t_polar);
    printf("McAdams SVD +avx                                %5.1fx\n",t_eigen_svd/t_mcadams_svd_avx);
    printf("McAdams R +avx                                  %5.1fx\n",t_eigen_svd/t_mcadams_avx);
    printf("Cayley Gershgorin                               %5.1fx\n",t_eigen_svd/t_cayley);
    printf("Cayley Gershgorin + avx                         %5.1fx\n",t_eigen_svd/t_cayley_avx);
    printf("Cayley Conservative + warm-start                %5.1fx\n",t_eigen_svd/t_cayley_warm_start);
    printf("Cayley Conservative + warm-start + avx          %5.1fx\n",t_eigen_svd/t_cayley_warm_start_avx);
    printf("Cayley (APD) + warm-start + avx                 %5.1fx\n",t_eigen_svd/t_apd_newton_avx);

    printf("\nSpeedup over McAdams SVD +avx\n");
    printf("-------------------------------\n");
    printf("Eigen SVD                                       %5.2fx\n",t_mcadams_svd_avx/t_eigen_svd);
    printf("Horn eig                                        %5.2fx\n",t_mcadams_svd_avx/t_horn_eig);
    printf("FA3R                                            %5.2fx\n",t_mcadams_svd_avx/t_fa3r);
    printf("MBCM                                            %5.2fx\n",t_mcadams_svd_avx/t_mbcm);
    printf("Polar Decomp +avx                               %5.2fx\n",t_mcadams_svd_avx/t_polar);
    printf("McAdams SVD +avx                                %5.2fx\n",t_mcadams_svd_avx/t_mcadams_svd_avx);
    printf("McAdams R +avx                                  %5.2fx\n",t_mcadams_svd_avx/t_mcadams_avx);
    printf("Cayley Gershgorin                               %5.2fx\n",t_mcadams_svd_avx/t_cayley);
    printf("Cayley Gershgorin + avx                         %5.2fx\n",t_mcadams_svd_avx/t_cayley_avx);
    printf("Cayley Conservative + warm-start                %5.2fx\n",t_mcadams_svd_avx/t_cayley_warm_start);
    printf("Cayley Conservative + warm-start + avx          %5.2fx\n",t_mcadams_svd_avx/t_cayley_warm_start_avx);
    printf("Cayley (APD) + warm-start + avx                 %5.2fx\n",t_mcadams_svd_avx/t_apd_newton_avx);


    return EXIT_SUCCESS;

}
