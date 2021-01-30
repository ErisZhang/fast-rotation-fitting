
#ifndef __FA3R_H__
#define __FA3R_H__

#pragma once

#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;


void FA3R(Matrix3f * sigma,
          		   int num,
		   Matrix3f * rRes,
		   Vector3f * tRes);

#endif