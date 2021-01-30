#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <chrono>

#include <cstdlib>
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <stdlib.h>
#include <fstream>

#include <x86intrin.h> //avx
#include <float.h>


using namespace Eigen;
using namespace std;


int fit_rotation_small_no_avx(float* Mf, float* Rf, int num_of_group)
{

    float M1, M2, M3, M4, M5, M6, M7, M8, M9;
    float m1, m2, m3, m4, m5, m6, m7, m8, m9;
    float ro1, ro2, ro3, ro4, ro5, ro6, ro7, ro8, ro9;
    float d1, d2, d3;
    float a1, a2, a3, b1, b2, b3, c1, c2, c3;
    float denom;
    float z1, z2, z3;
    float rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8, rn9;
    float c;

    int idx, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9;

    for (int i = 0; i < num_of_group; i++) {
        // inner loop is for each element of the grouped 8 matrices
        // this will be transformed into __mm256 later
        for (int j = 0; j < 8; j++) {

            idx = i*72+j;

            idx1 = idx+0*8;
            idx2 = idx+1*8;
            idx3 = idx+2*8;
            idx4 = idx+3*8;
            idx5 = idx+4*8;
            idx6 = idx+5*8;
            idx7 = idx+6*8;
            idx8 = idx+7*8;
            idx9 = idx+8*8;

            // Row major naming convention
            M1 = Mf[idx1]; // M(0,0)
            M2 = Mf[idx2]; // M(0,1)
            M3 = Mf[idx3]; // M(0,2)
            M4 = Mf[idx4]; // M(1,0)
            M5 = Mf[idx5]; // M(1,1)
            M6 = Mf[idx6]; // M(1,2)
            M7 = Mf[idx7]; // M(2,0)
            M8 = Mf[idx8]; // M(2,1)
            M9 = Mf[idx9]; // M(2,2)

            ro1 = Rf[idx1]; // R(0,0)
            ro2 = Rf[idx2]; // R(0,1)
            ro3 = Rf[idx3]; // R(0,2)
            ro4 = Rf[idx4]; // R(1,0)
            ro5 = Rf[idx5]; // R(1,1)
            ro6 = Rf[idx6]; // R(1,2)
            ro7 = Rf[idx7]; // R(2,0)
            ro8 = Rf[idx8]; // R(2,1)
            ro9 = Rf[idx9]; // R(2,2)

            // update m first
            m1 = ro1*M1+ro4*M4+ro7*M7;
            m2 = ro1*M2+ro4*M5+ro7*M8;
            m3 = ro1*M3+ro4*M6+ro7*M9;
            m4 = ro2*M1+ro5*M4+ro8*M7;
            m5 = ro2*M2+ro5*M5+ro8*M8;
            m6 = ro2*M3+ro5*M6+ro8*M9;
            m7 = ro3*M1+ro6*M4+ro9*M7;
            m8 = ro3*M2+ro6*M5+ro9*M8;
            m9 = ro3*M3+ro6*M6+ro9*M9;


            // RHS
            d1 = m8-m6;
            d2 = m3-m7;
            d3 = m4-m2;

            c = m1+m5+m9;
            c = std::sqrt(d1*d1+d2*d2+d3*d3+c*c);

            // LHS
            a1 = 2*m1-(m1+m5+m9)-c;
            a2 = m2+m4;
            a3 = m3+m7;
            b1 = m2+m4;
            b2 = 2*m5-(m1+m5+m9)-c;
            b3 = m6+m8;
            c1 = m3+m7;
            c2 = m6+m8;
            c3 = 2*m9-(m1+m5+m9)-c;


            denom = a1*(b2*c3-b3*c2)-a2*(b1*c3-b3*c1)+a3*(b1*c2-b2*c1);
            z1 =    d1*(b2*c3-b3*c2)-d2*(b1*c3-b3*c1)+d3*(b1*c2-b2*c1);
            z2 =   -d1*(a2*c3-a3*c2)+d2*(a1*c3-a3*c1)-d3*(a1*c2-a2*c1);
            z3 =    d1*(a2*b3-a3*b2)-d2*(a1*b3-a3*b1)+d3*(a1*b2-a2*b1);

            z1 = z1/denom;
            z2 = z2/denom;
            z3 = z3/denom;

            denom = 1+z1*z1+z2*z2+z3*z3;

            // compute entries for R_new
            rn1 = (z1*z1-z2*z2-z3*z3+1)/denom;  // R(0,0)
            rn2 = (2*z1*z2-2*z3)/denom;         // R(0,1)
            rn3 = (2*z1*z3+2*z2)/denom;         // R(0,2)
            rn4 = (2*z1*z2+2*z3)/denom;         // R(1,0)
            rn5 = (-z1*z1+z2*z2-z3*z3+1)/denom; // R(1,1)
            rn6 = (2*z2*z3-2*z1)/denom;         // R(1,2)
            rn7 = (2*z1*z3-2*z2)/denom;         // R(2,0)
            rn8 = (2*z1+2*z2*z3)/denom;;        // R(2,1)
            rn9 = (-z1*z1-z2*z2+z3*z3+1)/denom; // R(2,2)

            Rf[idx1] = ro1*rn1+ro2*rn2+ro3*rn3;  // R(0,0)
            Rf[idx2] = ro1*rn4+ro2*rn5+ro3*rn6;  // R(0,1)
            Rf[idx3] = ro1*rn7+ro2*rn8+ro3*rn9;  // R(0,2)
            Rf[idx4] = ro4*rn1+ro5*rn2+ro6*rn3;  // R(1,0)
            Rf[idx5] = ro4*rn4+ro5*rn5+ro6*rn6;  // R(1,1)
            Rf[idx6] = ro4*rn7+ro5*rn8+ro6*rn9;  // R(1,2)
            Rf[idx7] = ro7*rn1+ro8*rn2+ro9*rn3;  // R(2,0)
            Rf[idx8] = ro7*rn4+ro8*rn5+ro9*rn6;  // R(2,1)
            Rf[idx9] = ro7*rn7+ro8*rn8+ro9*rn9;  // R(2,2)

        }

    }

    return 0;
}



int fit_rotation_small_avx(float* Mf, float* Rf, int num_of_group)
{

    // AVX
    __m256 m1, m2, m3, m4, m5, m6, m7, m8, m9;
    __m256 M1, M2, M3, M4, M5, M6, M7, M8, M9;
    __m256 a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3;
    __m256 A1, A2, A3, B1, B2, B3, C1, C2, C3;
    __m256 denom;

    __m256 cf1, cf2, cf3, c;
    cf1 = _mm256_set1_ps(2.0);
    cf2 = _mm256_set1_ps(2.0);
    cf3 = _mm256_set1_ps(1.0);

    int idx, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9;

    for (int i = 0; i < num_of_group; i++) {

        idx = i*72;

        idx1 = idx+0*8;
        idx2 = idx+1*8;
        idx3 = idx+2*8;
        idx4 = idx+3*8;
        idx5 = idx+4*8;
        idx6 = idx+5*8;
        idx7 = idx+6*8;
        idx8 = idx+7*8;
        idx9 = idx+8*8;

        // transformed into __mm256
        M1 = _mm256_load_ps(&Mf[idx1]);
        M2 = _mm256_load_ps(&Mf[idx2]);
        M3 = _mm256_load_ps(&Mf[idx3]);
        M4 = _mm256_load_ps(&Mf[idx4]);
        M5 = _mm256_load_ps(&Mf[idx5]);
        M6 = _mm256_load_ps(&Mf[idx6]);
        M7 = _mm256_load_ps(&Mf[idx7]);
        M8 = _mm256_load_ps(&Mf[idx8]);
        M9 = _mm256_load_ps(&Mf[idx9]);

        A1 = _mm256_load_ps(&Rf[idx1]);
        A2 = _mm256_load_ps(&Rf[idx2]);
        A3 = _mm256_load_ps(&Rf[idx3]);
        B1 = _mm256_load_ps(&Rf[idx4]);
        B2 = _mm256_load_ps(&Rf[idx5]);
        B3 = _mm256_load_ps(&Rf[idx6]);
        C1 = _mm256_load_ps(&Rf[idx7]);
        C2 = _mm256_load_ps(&Rf[idx8]);
        C3 = _mm256_load_ps(&Rf[idx9]);


        m1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, M1), _mm256_mul_ps(B1, M4)), _mm256_mul_ps(C1, M7));
        m2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, M2), _mm256_mul_ps(B1, M5)), _mm256_mul_ps(C1, M8));
        m3 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, M3), _mm256_mul_ps(B1, M6)), _mm256_mul_ps(C1, M9));
        m4 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A2, M1), _mm256_mul_ps(B2, M4)), _mm256_mul_ps(C2, M7));
        m5 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A2, M2), _mm256_mul_ps(B2, M5)), _mm256_mul_ps(C2, M8));
        m6 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A2, M3), _mm256_mul_ps(B2, M6)), _mm256_mul_ps(C2, M9));
        m7 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A3, M1), _mm256_mul_ps(B3, M4)), _mm256_mul_ps(C3, M7));
        m8 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A3, M2), _mm256_mul_ps(B3, M5)), _mm256_mul_ps(C3, M8));
        m9 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A3, M3), _mm256_mul_ps(B3, M6)), _mm256_mul_ps(C3, M9));


        d1 = _mm256_sub_ps(m8, m6);
        d2 = _mm256_sub_ps(m3, m7);
        d3 = _mm256_sub_ps(m4, m2);

        c = _mm256_add_ps(_mm256_add_ps(m1, m5), m9);
        c = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(d1, d1), _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3))), _mm256_mul_ps(c, c)));


        a1 = _mm256_sub_ps(_mm256_sub_ps(m1, _mm256_add_ps(m5, m9)), c);
        a2 = _mm256_add_ps(m2, m4);
        a3 = _mm256_add_ps(m3, m7);
        b1 = _mm256_add_ps(m2, m4);
        b2 = _mm256_sub_ps(_mm256_sub_ps(m5, _mm256_add_ps(m1, m9)), c);
        b3 = _mm256_add_ps(m6, m8);
        c1 = _mm256_add_ps(m3, m7);
        c2 = _mm256_add_ps(m6, m8);
        c3 = _mm256_sub_ps(_mm256_sub_ps(m9, _mm256_add_ps(m1, m5)), c);



        // a1*(b2*c3-b3*c2)-a2*(b1*c3-b3*c1)+a3*(b1*c2-b2*c1)
        m4 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(a1, _mm256_sub_ps(_mm256_mul_ps(b2, c3), _mm256_mul_ps(b3, c2))),
                                        _mm256_mul_ps(a2, _mm256_sub_ps(_mm256_mul_ps(b1, c3), _mm256_mul_ps(b3, c1)))),
                                        _mm256_mul_ps(a3, _mm256_sub_ps(_mm256_mul_ps(b1, c2), _mm256_mul_ps(b2, c1))));
        // d1*(b2*c3-b3*c2)-d2*(b1*c3-b3*c1)+d3*(b1*c2-b2*c1)
        m1 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(d1, _mm256_sub_ps(_mm256_mul_ps(b2, c3), _mm256_mul_ps(b3, c2))),
                                        _mm256_mul_ps(d2, _mm256_sub_ps(_mm256_mul_ps(b1, c3), _mm256_mul_ps(b3, c1)))),
                                        _mm256_mul_ps(d3, _mm256_sub_ps(_mm256_mul_ps(b1, c2), _mm256_mul_ps(b2, c1))));
        //-d1*(a2*c3-a3*c2)+d2*(a1*c3-a3*c1)-d3*(a1*c2-a2*c1)
        m2 = _mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(d2, _mm256_sub_ps(_mm256_mul_ps(a1, c3), _mm256_mul_ps(a3, c1))),
                                        _mm256_mul_ps(d1, _mm256_sub_ps(_mm256_mul_ps(a2, c3), _mm256_mul_ps(a3, c2)))),
                                        _mm256_mul_ps(d3, _mm256_sub_ps(_mm256_mul_ps(a1, c2), _mm256_mul_ps(a2, c1))));                                
        // d1*(a2*b3-a3*b2)-d2*(a1*b3-a3*b1)+d3*(a1*b2-a2*b1)
        m3 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(d1, _mm256_sub_ps(_mm256_mul_ps(a2, b3), _mm256_mul_ps(a3, b2))),
                                        _mm256_mul_ps(d2, _mm256_sub_ps(_mm256_mul_ps(a1, b3), _mm256_mul_ps(a3, b1)))),
                                        _mm256_mul_ps(d3, _mm256_sub_ps(_mm256_mul_ps(a1, b2), _mm256_mul_ps(a2, b1))));


        d1 = _mm256_div_ps(m1, m4); // z1 = z1/denom;
        d2 = _mm256_div_ps(m2, m4); // z2 = z2/denom;
        d3 = _mm256_div_ps(m3, m4); // z3 = z3/denom;

        // denom = 1+z1*z1+z2*z2+z3*z3;
        denom = _mm256_add_ps(_mm256_add_ps(cf3, _mm256_mul_ps(d1, d1)), _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)));


        // compute entries for R_new
        m1 = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(d1, d1), _mm256_mul_ps(d2, d2)), _mm256_sub_ps(_mm256_mul_ps(d3, d3), cf3)), denom); // R(0,0)
        m2 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_sub_ps(_mm256_mul_ps(d1, d2), d3)), denom));                                                        // R(0,1)
        m3 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_add_ps(_mm256_mul_ps(d1, d3), d2)), denom));                                                        // R(0,2)
        m4 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_add_ps(_mm256_mul_ps(d1, d2), d3)), denom));                                                        // R(1,0)
        m5 = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d1, d1)), _mm256_sub_ps(_mm256_mul_ps(d3, d3), cf3)), denom); // R(1,1)
        m6 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_sub_ps(_mm256_mul_ps(d2, d3), d1)), denom));                                                        // R(1,2)
        m7 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_sub_ps(_mm256_mul_ps(d1, d3), d2)), denom));                                                        // R(2,0)
        m8 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_add_ps(_mm256_mul_ps(d2, d3), d1)), denom));                                                        // R(2,1)
        m9 = _mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(d3, d3), cf3), _mm256_add_ps(_mm256_mul_ps(d1, d1), _mm256_mul_ps(d2, d2))), denom); // R(2,2)


        _mm256_store_ps(&Rf[idx1], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, m1), _mm256_mul_ps(A2, m2)), _mm256_mul_ps(A3, m3)));
        _mm256_store_ps(&Rf[idx2], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, m4), _mm256_mul_ps(A2, m5)), _mm256_mul_ps(A3, m6)));
        _mm256_store_ps(&Rf[idx3], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(A1, m7), _mm256_mul_ps(A2, m8)), _mm256_mul_ps(A3, m9)));
        _mm256_store_ps(&Rf[idx4], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(B1, m1), _mm256_mul_ps(B2, m2)), _mm256_mul_ps(B3, m3)));
        _mm256_store_ps(&Rf[idx5], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(B1, m4), _mm256_mul_ps(B2, m5)), _mm256_mul_ps(B3, m6)));
        _mm256_store_ps(&Rf[idx6], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(B1, m7), _mm256_mul_ps(B2, m8)), _mm256_mul_ps(B3, m9)));
        _mm256_store_ps(&Rf[idx7], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(C1, m1), _mm256_mul_ps(C2, m2)), _mm256_mul_ps(C3, m3)));
        _mm256_store_ps(&Rf[idx8], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(C1, m4), _mm256_mul_ps(C2, m5)), _mm256_mul_ps(C3, m6)));
        _mm256_store_ps(&Rf[idx9], _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(C1, m7), _mm256_mul_ps(C2, m8)), _mm256_mul_ps(C3, m9)));


    }

    return 0;
}



int fit_rotation_no_avx(float* Mf, float* Rf, int num_of_group)
{

    float M1, M2, M3, M4, M5, M6, M7, M8, M9;
    float m1, m2, m3, m4, m5, m6, m7, m8, m9;
    float m1_, m2_, m3_, m4_, m5_, m6_, m7_, m8_, m9_;
    float d1, d2, d3;
    float a1, a2, a3, b1, b2, b3, c1, c2, c3;
    float denom;
    float z1, z2, z3;
    float r1, r2, r3, r4, r5, r6, r7, r8, r9;
    float r1_, r2_, r3_, r4_, r5_, r6_, r7_, r8_, r9_;
    float rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8, rn9;


    int idx, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9;

    for (int i = 0; i < num_of_group; i++) {
        // inner loop is for each element of the grouped 8 matrices
        // this will be transformed into __mm256 later
        for (int j = 0; j < 8; j++) {

            // compute indices
            idx = i*72+j;
            idx1 = idx+0*8;
            idx2 = idx+1*8;
            idx3 = idx+2*8;
            idx4 = idx+3*8;
            idx5 = idx+4*8;
            idx6 = idx+5*8;
            idx7 = idx+6*8;
            idx8 = idx+7*8;
            idx9 = idx+8*8;

            // Row major naming convention
            M1 = Mf[idx1]; // M(0,0)
            M2 = Mf[idx2]; // M(0,1)
            M3 = Mf[idx3]; // M(0,2)
            M4 = Mf[idx4]; // M(1,0)
            M5 = Mf[idx5]; // M(1,1)
            M6 = Mf[idx6]; // M(1,2)
            M7 = Mf[idx7]; // M(2,0)
            M8 = Mf[idx8]; // M(2,1)
            M9 = Mf[idx9]; // M(2,2)


            // rotation matrix
            r1 = Rf[idx1];
            r2 = Rf[idx2];
            r3 = Rf[idx3];
            r4 = Rf[idx4];
            r5 = Rf[idx5];
            r6 = Rf[idx6];
            r7 = Rf[idx7];
            r8 = Rf[idx8];
            r9 = Rf[idx9];


            // update m then
            // M = M * Q;
            m1 = M1*r1+M2*r4+M3*r7;
            m2 = M1*r2+M2*r5+M3*r8;
            m3 = M1*r3+M2*r6+M3*r9;
            m4 = M4*r1+M5*r4+M6*r7;
            m5 = M4*r2+M5*r5+M6*r8;
            m6 = M4*r3+M5*r6+M6*r9;
            m7 = M7*r1+M8*r4+M9*r7;
            m8 = M7*r2+M8*r5+M9*r8;
            m9 = M7*r3+M8*r6+M9*r9;


            for (int k = 0; k < 500; k++) {

                // ----------------------------------------- //
                // RHS
                d1 = m8-m6;
                d2 = m3-m7;
                d3 = m4-m2;

                float c = m1+m5+m9;

                float l0 = 2*m1 + std::abs(m2+m4) + std::abs(m3+m7);
                float l1 = std::abs(m4+m2) + 2*m5 + std::abs(m6+m8);
                float l2 = std::abs(m7+m3) + std::abs(m8+m6) + 2*m9;
                float l = std::max(l0, std::max(l1,l2)) - 2 * c;

                if (l > 0) c = c + l;

                c = std::sqrt(c * c + d1 * d1 + d2 * d2 + d3 * d3);

                // LHS
                a1 = 2*m1-(m1+m5+m9)-c;
                a2 = m2+m4;
                a3 = m3+m7;
                b1 = m2+m4;
                b2 = 2*m5-(m1+m5+m9)-c;
                b3 = m6+m8;
                c1 = m3+m7;
                c2 = m6+m8;
                c3 = 2*m9-(m1+m5+m9)-c;


                denom = a1*(b2*c3-b3*c2)-a2*(b1*c3-b3*c1)+a3*(b1*c2-b2*c1);
                z1 =    d1*(b2*c3-b3*c2)-d2*(b1*c3-b3*c1)+d3*(b1*c2-b2*c1);
                z2 =   -d1*(a2*c3-a3*c2)+d2*(a1*c3-a3*c1)-d3*(a1*c2-a2*c1);
                z3 =    d1*(a2*b3-a3*b2)-d2*(a1*b3-a3*b1)+d3*(a1*b2-a2*b1);

                z1 = z1/denom;
                z2 = z2/denom;
                z3 = z3/denom;

                denom = 1+z1*z1+z2*z2+z3*z3;

                // compute entries for R_new
                rn1 = (z1*z1-z2*z2-z3*z3+1)/denom;  // R_new(0,0)
                rn2 = (2*z1*z2-2*z3)/denom;         // R_new(0,1)
                rn3 = (2*z1*z3+2*z2)/denom;         // R_new(0,2)
                rn4 = (2*z1*z2+2*z3)/denom;         // R_new(1,0)
                rn5 = (-z1*z1+z2*z2-z3*z3+1)/denom; // R_new(1,1)
                rn6 = (2*z2*z3-2*z1)/denom;         // R_new(1,2)
                rn7 = (2*z1*z3-2*z2)/denom;         // R_new(2,0)
                rn8 = (2*z1+2*z2*z3)/denom;;        // R_new(2,1)
                rn9 = (-z1*z1-z2*z2+z3*z3+1)/denom; // R_new(2,2)

                // save the value for R
                r1_ = r1;
                r2_ = r2;
                r3_ = r3;
                r4_ = r4;
                r5_ = r5;
                r6_ = r6;
                r7_ = r7;
                r8_ = r8;
                r9_ = r9;

                r1 = r1_*rn1+r2_*rn4+r3_*rn7;  // R(0,0)
                r2 = r1_*rn2+r2_*rn5+r3_*rn8;  // R(0,1)
                r3 = r1_*rn3+r2_*rn6+r3_*rn9;  // R(0,2)
                r4 = r4_*rn1+r5_*rn4+r6_*rn7;  // R(1,0)
                r5 = r4_*rn2+r5_*rn5+r6_*rn8;  // R(1,1)
                r6 = r4_*rn3+r5_*rn6+r6_*rn9;  // R(1,2)
                r7 = r7_*rn1+r8_*rn4+r9_*rn7;  // R(2,0)
                r8 = r7_*rn2+r8_*rn5+r9_*rn8;  // R(2,1)
                r9 = r7_*rn3+r8_*rn6+r9_*rn9;  // R(2,2)

                // convergence criterion
                float s = z1 * z1 + z2 * z2 + z3 * z3;
                if (s < 1e-8) {
                    break;
                }


                m1_ = m1;
                m2_ = m2;
                m3_ = m3;
                m4_ = m4;
                m5_ = m5;
                m6_ = m6;
                m7_ = m7;
                m8_ = m8;
                m9_ = m9;

                // update M
                m1 = m1_*rn1+m2_*rn4+m3_*rn7;  // M(0,0)
                m2 = m1_*rn2+m2_*rn5+m3_*rn8;  // M(0,1)
                m3 = m1_*rn3+m2_*rn6+m3_*rn9;  // M(0,2)
                m4 = m4_*rn1+m5_*rn4+m6_*rn7;  // M(1,0)
                m5 = m4_*rn2+m5_*rn5+m6_*rn8;  // M(1,1)
                m6 = m4_*rn3+m5_*rn6+m6_*rn9;  // M(1,2)
                m7 = m7_*rn1+m8_*rn4+m9_*rn7;  // M(2,0)
                m8 = m7_*rn2+m8_*rn5+m9_*rn8;  // M(2,1)
                m9 = m7_*rn3+m8_*rn6+m9_*rn9;  // M(2,2)
            }

            // take the transpose
            Rf[idx1] = r1;  // R(0,0)
            Rf[idx2] = r4;  // R(0,1)
            Rf[idx3] = r7;  // R(0,2)
            Rf[idx4] = r2;  // R(1,0)
            Rf[idx5] = r5;  // R(1,1)
            Rf[idx6] = r8;  // R(1,2)
            Rf[idx7] = r3;  // R(2,0)
            Rf[idx8] = r6;  // R(2,1)
            Rf[idx9] = r9;  // R(2,2)

        }
    }

    return 0;
}





int fit_rotation_avx(float* Mf, float* Rf, int num_of_group)
{

   // AVX
    __m256 m1, m2, m3, m4, m5, m6, m7, m8, m9;
    __m256 m1_, m2_, m3_, m4_, m5_, m6_, m7_, m8_, m9_;
    __m256 M1, M2, M3, M4, M5, M6, M7, M8, M9;
    __m256 r1, r2, r3, r4, r5, r6, r7, r8, r9;
    __m256 r1_, r2_, r3_, r4_, r5_, r6_, r7_, r8_, r9_;
    __m256 rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8, rn9;
    __m256 a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3;
    __m256 z1, z2, z3;
    __m256 denom, s;


    __m256 cf1, cf2, cf3, c;
    cf1 = _mm256_set1_ps(2.0);
    cf2 = _mm256_set1_ps(2.0);
    cf3 = _mm256_set1_ps(1.0);

    __m256 l0, l1, l2, l_0, l_1, l_2, l;
    __m256 trace, sum;
    __m256 vcmp;


    int idx, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9;

    for (int i = 0; i < num_of_group; i++) {

        // compute indices
        idx = i*72;
        idx1 = idx+0*8;
        idx2 = idx+1*8;
        idx3 = idx+2*8;
        idx4 = idx+3*8;
        idx5 = idx+4*8;
        idx6 = idx+5*8;
        idx7 = idx+6*8;
        idx8 = idx+7*8;
        idx9 = idx+8*8;

        // transformed into __mm256
        M1 = _mm256_load_ps(&Mf[idx1]);
        M2 = _mm256_load_ps(&Mf[idx2]);
        M3 = _mm256_load_ps(&Mf[idx3]);
        M4 = _mm256_load_ps(&Mf[idx4]);
        M5 = _mm256_load_ps(&Mf[idx5]);
        M6 = _mm256_load_ps(&Mf[idx6]);
        M7 = _mm256_load_ps(&Mf[idx7]);
        M8 = _mm256_load_ps(&Mf[idx8]);
        M9 = _mm256_load_ps(&Mf[idx9]);

        // initial rotation matrix
        r1 = _mm256_load_ps(&Rf[idx1]);
        r2 = _mm256_load_ps(&Rf[idx2]);
        r3 = _mm256_load_ps(&Rf[idx3]);
        r4 = _mm256_load_ps(&Rf[idx4]);
        r5 = _mm256_load_ps(&Rf[idx5]);
        r6 = _mm256_load_ps(&Rf[idx6]);
        r7 = _mm256_load_ps(&Rf[idx7]);
        r8 = _mm256_load_ps(&Rf[idx8]);
        r9 = _mm256_load_ps(&Rf[idx9]);


        // update m then
        // M = M * Q;
        m1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M1, r1), _mm256_mul_ps(M2, r4)), _mm256_mul_ps(M3, r7));
        m2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M1, r2), _mm256_mul_ps(M2, r5)), _mm256_mul_ps(M3, r8));
        m3 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M1, r3), _mm256_mul_ps(M2, r6)), _mm256_mul_ps(M3, r9));
        m4 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M4, r1), _mm256_mul_ps(M5, r4)), _mm256_mul_ps(M6, r7));
        m5 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M4, r2), _mm256_mul_ps(M5, r5)), _mm256_mul_ps(M6, r8));
        m6 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M4, r3), _mm256_mul_ps(M5, r6)), _mm256_mul_ps(M6, r9));
        m7 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M7, r1), _mm256_mul_ps(M8, r4)), _mm256_mul_ps(M9, r7));
        m8 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M7, r2), _mm256_mul_ps(M8, r5)), _mm256_mul_ps(M9, r8));
        m9 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(M7, r3), _mm256_mul_ps(M8, r6)), _mm256_mul_ps(M9, r9));


        for (int k = 0; k < 500; k++) {

            // previous
            d1 = _mm256_sub_ps(m8, m6);
            d2 = _mm256_sub_ps(m3, m7);
            d3 = _mm256_sub_ps(m4, m2);

            trace = _mm256_add_ps(m1, _mm256_add_ps(m5, m9));

            l_0 = _mm256_add_ps(m2, m4);
            l_1 = _mm256_add_ps(m3, m7);
            l_2 = _mm256_add_ps(m6, m8);
            l0 = _mm256_add_ps(_mm256_mul_ps(cf1, m1), _mm256_add_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), l_0), l_0), _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), l_1), l_1)));
            l1 = _mm256_add_ps(_mm256_mul_ps(cf1, m5), _mm256_add_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), l_0), l_0), _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), l_2), l_2)));
            l2 = _mm256_add_ps(_mm256_mul_ps(cf1, m9), _mm256_add_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), l_1), l_1), _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), l_2), l_2)));
            l = _mm256_max_ps(l0, _mm256_max_ps(l1, l2)) - _mm256_mul_ps(cf2, trace);

            c = _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), l), trace);

            sum = _mm256_add_ps(_mm256_mul_ps(d1, d1), _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)));
            c = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(c, c), sum));


            a1 = _mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(cf1, m1), trace), c);
            a2 = _mm256_add_ps(m2, m4);
            a3 = _mm256_add_ps(m3, m7);
            b1 = _mm256_add_ps(m2, m4);
            b2 = _mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(cf1, m5), trace), c);
            b3 = _mm256_add_ps(m6, m8);
            c1 = _mm256_add_ps(m3, m7);
            c2 = _mm256_add_ps(m6, m8);
            c3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(cf1, m9), trace), c);



            // a1*(b2*c3-b3*c2)-a2*(b1*c3-b3*c1)+a3*(b1*c2-b2*c1)
            denom = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(a1, _mm256_sub_ps(_mm256_mul_ps(b2, c3), _mm256_mul_ps(b3, c2))),
                                            _mm256_mul_ps(a2, _mm256_sub_ps(_mm256_mul_ps(b1, c3), _mm256_mul_ps(b3, c1)))),
                                            _mm256_mul_ps(a3, _mm256_sub_ps(_mm256_mul_ps(b1, c2), _mm256_mul_ps(b2, c1))));
            // d1*(b2*c3-b3*c2)-d2*(b1*c3-b3*c1)+d3*(b1*c2-b2*c1)
            z1 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(d1, _mm256_sub_ps(_mm256_mul_ps(b2, c3), _mm256_mul_ps(b3, c2))),
                                            _mm256_mul_ps(d2, _mm256_sub_ps(_mm256_mul_ps(b1, c3), _mm256_mul_ps(b3, c1)))),
                                            _mm256_mul_ps(d3, _mm256_sub_ps(_mm256_mul_ps(b1, c2), _mm256_mul_ps(b2, c1))));
            //-d1*(a2*c3-a3*c2)+d2*(a1*c3-a3*c1)-d3*(a1*c2-a2*c1)
            z2 = _mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(d2, _mm256_sub_ps(_mm256_mul_ps(a1, c3), _mm256_mul_ps(a3, c1))),
                                            _mm256_mul_ps(d1, _mm256_sub_ps(_mm256_mul_ps(a2, c3), _mm256_mul_ps(a3, c2)))),
                                            _mm256_mul_ps(d3, _mm256_sub_ps(_mm256_mul_ps(a1, c2), _mm256_mul_ps(a2, c1))));                                
            // d1*(a2*b3-a3*b2)-d2*(a1*b3-a3*b1)+d3*(a1*b2-a2*b1)
            z3 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(d1, _mm256_sub_ps(_mm256_mul_ps(a2, b3), _mm256_mul_ps(a3, b2))),
                                            _mm256_mul_ps(d2, _mm256_sub_ps(_mm256_mul_ps(a1, b3), _mm256_mul_ps(a3, b1)))),
                                            _mm256_mul_ps(d3, _mm256_sub_ps(_mm256_mul_ps(a1, b2), _mm256_mul_ps(a2, b1))));


            z1 = _mm256_div_ps(z1, denom); // z1 = z1/denom;
            z2 = _mm256_div_ps(z2, denom); // z2 = z2/denom;
            z3 = _mm256_div_ps(z3, denom); // z3 = z3/denom;

            // denom = 1+z1*z1+z2*z2+z3*z3;
            s = _mm256_add_ps(_mm256_mul_ps(z1, z1), _mm256_add_ps(_mm256_mul_ps(z2, z2), _mm256_mul_ps(z3, z3)));
            denom = _mm256_add_ps(cf3, s);

            // compute entries for R_new
            rn1 = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(z1, z1), _mm256_mul_ps(z2, z2)), _mm256_sub_ps(_mm256_mul_ps(z3, z3), cf3)), denom); // R(0,0)
            rn2 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_sub_ps(_mm256_mul_ps(z1, z2), z3)), denom));                                                        // R(0,1)
            rn3 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_add_ps(_mm256_mul_ps(z1, z3), z2)), denom));                                                        // R(0,2)
            rn4 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_add_ps(_mm256_mul_ps(z1, z2), z3)), denom));                                                        // R(1,0)
            rn5 = _mm256_div_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_mul_ps(z2, z2), _mm256_mul_ps(z1, z1)), _mm256_sub_ps(_mm256_mul_ps(z3, z3), cf3)), denom); // R(1,1)
            rn6 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_sub_ps(_mm256_mul_ps(z2, z3), z1)), denom));                                                        // R(1,2)
            rn7 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_sub_ps(_mm256_mul_ps(z1, z3), z2)), denom));                                                        // R(2,0)
            rn8 = _mm256_mul_ps(cf1, _mm256_div_ps((_mm256_add_ps(_mm256_mul_ps(z2, z3), z1)), denom));                                                        // R(2,1)
            rn9 = _mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(z3, z3), cf3), _mm256_add_ps(_mm256_mul_ps(z1, z1), _mm256_mul_ps(z2, z2))), denom); // R(2,2)

            // save R
            r1_ = r1;
            r2_ = r2;
            r3_ = r3;
            r4_ = r4;
            r5_ = r5;
            r6_ = r6;
            r7_ = r7;
            r8_ = r8;
            r9_ = r9;

            // update R
            r1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r1_, rn1), _mm256_mul_ps(r2_, rn4)), _mm256_mul_ps(r3_, rn7));
            r2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r1_, rn2), _mm256_mul_ps(r2_, rn5)), _mm256_mul_ps(r3_, rn8));
            r3 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r1_, rn3), _mm256_mul_ps(r2_, rn6)), _mm256_mul_ps(r3_, rn9));
            r4 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r4_, rn1), _mm256_mul_ps(r5_, rn4)), _mm256_mul_ps(r6_, rn7));
            r5 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r4_, rn2), _mm256_mul_ps(r5_, rn5)), _mm256_mul_ps(r6_, rn8));
            r6 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r4_, rn3), _mm256_mul_ps(r5_, rn6)), _mm256_mul_ps(r6_, rn9));
            r7 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r7_, rn1), _mm256_mul_ps(r8_, rn4)), _mm256_mul_ps(r9_, rn7));
            r8 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r7_, rn2), _mm256_mul_ps(r8_, rn5)), _mm256_mul_ps(r9_, rn8));
            r9 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r7_, rn3), _mm256_mul_ps(r8_, rn6)), _mm256_mul_ps(r9_, rn9));


            vcmp = _mm256_cmp_ps(s, _mm256_set1_ps(1e-15), _CMP_LT_OQ);
            int cmp = _mm256_movemask_ps(vcmp);
            if (cmp == 0xFF) {
                break;
            }


            // save M
            m1_ = m1;
            m2_ = m2;
            m3_ = m3;
            m4_ = m4;
            m5_ = m5;
            m6_ = m6;
            m7_ = m7;
            m8_ = m8;
            m9_ = m9;

            // update M
            m1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m1_, rn1), _mm256_mul_ps(m2_, rn4)), _mm256_mul_ps(m3_, rn7));
            m2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m1_, rn2), _mm256_mul_ps(m2_, rn5)), _mm256_mul_ps(m3_, rn8));
            m3 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m1_, rn3), _mm256_mul_ps(m2_, rn6)), _mm256_mul_ps(m3_, rn9));
            m4 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m4_, rn1), _mm256_mul_ps(m5_, rn4)), _mm256_mul_ps(m6_, rn7));
            m5 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m4_, rn2), _mm256_mul_ps(m5_, rn5)), _mm256_mul_ps(m6_, rn8));
            m6 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m4_, rn3), _mm256_mul_ps(m5_, rn6)), _mm256_mul_ps(m6_, rn9));
            m7 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m7_, rn1), _mm256_mul_ps(m8_, rn4)), _mm256_mul_ps(m9_, rn7));
            m8 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m7_, rn2), _mm256_mul_ps(m8_, rn5)), _mm256_mul_ps(m9_, rn8));
            m9 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(m7_, rn3), _mm256_mul_ps(m8_, rn6)), _mm256_mul_ps(m9_, rn9));

        }

        // take the transpose of R
        _mm256_store_ps(&Rf[idx1], r1);
        _mm256_store_ps(&Rf[idx2], r4);
        _mm256_store_ps(&Rf[idx3], r7);
        _mm256_store_ps(&Rf[idx4], r2);
        _mm256_store_ps(&Rf[idx5], r5);
        _mm256_store_ps(&Rf[idx6], r8);
        _mm256_store_ps(&Rf[idx7], r3);
        _mm256_store_ps(&Rf[idx8], r6);
        _mm256_store_ps(&Rf[idx9], r9);
    }

    return 0;

}

