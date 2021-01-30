#include <Eigen/Core>

int fit_rotation_small_no_avx(
    float* Mf,
    float* Rf,
    int num_of_group);


int fit_rotation_small_avx(
    float* Mf,
    float* Rf,
    int num_of_group);


int fit_rotation_no_avx(
    float* Mf,
    float* Rf,
    int num_of_group);


int fit_rotation_avx(
    float* Mf,
    float* Rf,
    int num_of_group);