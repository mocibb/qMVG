/*!
 * Author: mocibb mocibb@163.com
 * Group:  CarrotSLAM https://github.com/mocibb/CarrotSLAM
 * Name:   mvg_motion.h
 * Date:   2015.11.01
 * Func:   decompose rigid motion from 2-view geometry.
 *
 *
 * The MIT License (MIT)
 * Copyright (c) 2015 CarrotSLAM Group
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef CORE_MVG_MOTION_H_
#define CORE_MVG_MOTION_H_

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include "mvg.h"

namespace carrotslam {

void essentialFromMotion(Eigen::Matrix3f &E, const Eigen::Matrix3f &R, const Eigen::Vector3f &t) {
    Eigen::Vector3f t_norm = t.normalized();
    Eigen::Matrix3f t_cross;
    t_cross << 0, -t_norm[2], t_norm[1], t_norm[2], 0, -t_norm[0], -t_norm[1], t_norm[0], 0;
    E = t_cross * R;
}

/*!
 *
 */
bool decomposeEssential(const Eigen::Matrix3f &E, const Eigen::Matrix3f &K1,
                        const Eigen::MatrixXf *x1,
                        const Eigen::Matrix3f &K2,
                        const Eigen::MatrixXf *x2,
                        Eigen::Matrix3f &R, Eigen::Vector3f &t) {
  std::vector<Eigen::Matrix3f> Rs;
  std::vector<Eigen::Vector3f> ts;

  Eigen::JacobiSVD<Eigen::Matrix3f> USV(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = USV.matrixU();
  Eigen::Matrix3f Vt = USV.matrixV().transpose();

  // Last column of U is undetermined since d = (a a 0).
  if (U.determinant() < 0) {
    U.col(2) *= -1;
  }
  // Last row of Vt is undetermined since d = (a a 0).
  if (Vt.determinant() < 0) {
    Vt.row(2) *= -1;
  }

  Eigen::Matrix3f W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  Eigen::Matrix3f U_W_Vt = U * W * Vt;
  Eigen::Matrix3f U_Wt_Vt = U * W.transpose() * Vt;

  Rs.push_back(U_W_Vt);
  Rs.push_back(U_W_Vt);
  Rs.push_back(U_Wt_Vt);
  Rs.push_back(U_Wt_Vt);
  ts.push_back(U.col(2));
  ts.push_back(-U.col(2));
  ts.push_back(U.col(2));
  ts.push_back(-U.col(2));

  MatrixP P1 = MatrixP::Zero();
  P1.block<3, 3>(0, 0) = Eigen::Vector3f::Ones().asDiagonal();
  P1 = K1 * P1;

  Eigen::Vector4i posCnt = Eigen::Vector4i::Zero();
  for (int i = 0; i < 4; ++i) {
    MatrixP P2;
    Eigen::Vector3f X;
    P2.block<3, 3>(0, 0) = Rs[i];
    P2.block<3, 1>(0, 3) = ts[i];
    P2 = K2 * P2;
    for (int j = 0; j < x1->cols(); j++) {
      triangulate(P1, x1->col(j), P2, x2->col(j), X);

      // Test if point is front to the two cameras (positive depth)
      if (X[2] > 0 && (Rs[i] * X + ts[i])[2] > 0) {
        posCnt[i]++;
      }
    }
  }

  typename Eigen::Vector4i::Index solution;
  int cnt = posCnt.maxCoeff(&solution);

  if (cnt > 0) {
    R = Rs[solution];
    t = ts[solution];

    return true;
  }

  return false;
}


} // namespace carrotslam
#endif  /* CORE_MVG_MOTION_H_ */
