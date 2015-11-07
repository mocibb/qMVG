/*!
 * Author: mocibb mocibb@163.com
 * Group:  CarrotSLAM https://github.com/mocibb/CarrotSLAM
 * Name:   mvg.cpp
 * Date:   2015.10.31
 * Func:   implement methods in 2-view geometry.
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
#include "mvg.h"


using namespace std;
using namespace Eigen;

namespace carrotslam {

/*!
 * y = H x
 */
void FourPointHomographySolver::solve(const MatrixXf &x, const MatrixXf &y,
                                      vector<Matrix3f> *Hs) {
  const int N = x.cols();

  //对应点数目不够，需要满足A的rank为8
  if (N < 4) {
    throw std::runtime_error("not enough point for compute homograph!");
  }

  typedef Matrix<float, Dynamic, 9> MatA;
  MatA A = MatA::Zero(2 * N, 9);

  //h11 h21 h31 h12...
  for (int i = 0; i < N; i++) {
    A(2 * i, 0) = x(0, i);
    A(2 * i, 3) = x(1, i);
    A(2 * i, 6) = 1;
    A(2 * i, 2) = -y(0, i) * x(0, i);
    A(2 * i, 5) = -y(0, i) * x(1, i);
    A(2 * i, 8) = -y(0, i);
    A(2 * i + 1, 1) = x(0, i);
    A(2 * i + 1, 4) = x(1, i);
    A(2 * i + 1, 7) = 1;
    A(2 * i + 1, 2) = -y(1, i) * x(0, i);
    A(2 * i + 1, 5) = -y(1, i) * x(1, i);
    A(2 * i + 1, 8) = -y(1, i);
  }

  Matrix<float, 9, 1> h;
  nullspace<MatA, Matrix<float, 9, 1>, 8>(A, h, false);
  Hs->push_back(Matrix3f(h.data()));
}

/*!
 * x1' F x2 = 0
 */
void EightPointFundamentalSolver::solve(const MatrixXf &x1, const MatrixXf &x2,
                                        vector<Matrix3f> *Fs) {
  const int N = x1.cols();

  //对应点数目不够，需要满足A的rank为8
  if (N < 8) {
    throw std::runtime_error("not enough point for compute fundamental!");
  }

  typedef Matrix<float, Dynamic, 9> MatA;
  MatA A = MatA::Zero(N, 9);

  //f11 f21 f31
  for (int i = 0; i < N; i++) {
    A(i, 0) = x1(0, i) * x2(0, i);
    A(i, 3) = x1(0, i) * x2(1, i);
    A(i, 6) = x1(0, i);
    A(i, 1) = x1(1, i) * x2(0, i);
    A(i, 4) = x1(1, i) * x2(1, i);
    A(i, 7) = x1(1, i);
    A(i, 2) = x2(0, i);
    A(i, 5) = x2(1, i);
    A(i, 8) = 1;
  }

  Matrix<float, 9, 1> f;
  nullspace<>(A, f, false);

  Matrix3f F_raw = Map<Matrix3f>(f.data());
  JacobiSVD<Matrix3f> svdF(F_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Vector3f d = svdF.singularValues();

  d[2] = 0.0;
  Fs->push_back(svdF.matrixU() * d.asDiagonal() * svdF.matrixV().transpose());

}

/*!
 * x1' E x2 = 0
 */
void EightPointEssentialSolver::solve(const MatrixXf &x1, const MatrixXf &x2,
                                        vector<Matrix3f> *Es) {
  const int N = x1.cols();

  //对应点数目不够，需要满足A的rank为8
  if (N < 8) {
    throw std::runtime_error("not enough point for compute fundamental!");
  }

  typedef Matrix<float, Dynamic, 9> MatA;
  MatA A = MatA::Zero(N, 9);

  //e11 e21 e31
  for (int i = 0; i < N; i++) {
    A(i, 0) = x1(0, i) * x2(0, i);
    A(i, 3) = x1(0, i) * x2(1, i);
    A(i, 6) = x1(0, i);
    A(i, 1) = x1(1, i) * x2(0, i);
    A(i, 4) = x1(1, i) * x2(1, i);
    A(i, 7) = x1(1, i);
    A(i, 2) = x2(0, i);
    A(i, 5) = x2(1, i);
    A(i, 8) = 1;
  }

  Matrix<float, 9, 1> f;
  nullspace<>(A, f, false);

  Matrix3f F_raw = Map<Matrix3f>(f.data());
  JacobiSVD<Matrix3f> svdF(F_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Vector3f sing = svdF.singularValues();
  //page 294 in HZ Result 11.1.
  Vector3f diag((sing[0]+sing[1])/2., (sing[0]+sing[1])/2., 0);
  Es->push_back(svdF.matrixU() * diag.asDiagonal() * svdF.matrixV().transpose());

}

void triangulate(const MatrixP &P1, const Eigen::Vector2f &x1,
                 const MatrixP &P2, const Eigen::Vector2f &x2,
                 Eigen::Vector3f& X) {
  Eigen::Matrix4f A;
  for (int i = 0; i < 4; ++i) {
    A(0, i) = x1[0] * P1(2, i) - P1(0, i);
    A(1, i) = x1[1] * P1(2, i) - P1(1, i);
    A(2, i) = x2[0] * P2(2, i) - P2(0, i);
    A(3, i) = x2[1] * P2(2, i) - P2(1, i);
  }
  Eigen::Vector4f Xh;
  nullspace<>(A, Xh, false);
  X = Xh.head<3>() / Xh[3];
}

} // namespace carrotslam


