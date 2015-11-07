/*!
 * Author: mocibb mocibb@163.com
 * Group:  CarrotSLAM https://github.com/mocibb/CarrotSLAM
 * Name:   mvg.h
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
#ifndef CORE_MVG_H_
#define CORE_MVG_H_

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>

namespace carrotslam {

template<typename TMat, typename TCols>
TMat extractColumns(const TMat &A, const TCols *columns) {
  TMat compressed(A.rows(), columns->size());
  for (size_t i = 0; i < static_cast<size_t>(columns->size()); ++i) {
    compressed.col(i) = A.col((*columns)[i]);
  }
  return compressed;
}

/*！
 * Homograph矩阵的反归一化方法 参考 HZ page 282.
 */
struct UnnormalizerH {
  // Denormalize method used in Homograph. See HZ page 109.
  static void unnormalize(const Eigen::Matrix3f &T1, const Eigen::Matrix3f &T2,
                          Eigen::Matrix3f &H) {
    H = T2.inverse() * H * T1;
  }
};

/*！
 * Essential矩阵的反归一化方法 参考 HZ page 282.
 */
struct UnnormalizerE {
  // Denormalize method used in Essential. See HZ page 282.
  static void unnormalize(const Eigen::Matrix3f &T1, const Eigen::Matrix3f &T2,
                          Eigen::Matrix3f &H) {
    H = T2.transpose() * H * T1;
  }
};

/*！
 *openMVG中采用 u,v ~ N([0,0]^t, diag(sqrt(2), sqrt(2))),
 *     这里采用 u,v ~ N([1,1]^t, diag(1, 1))
 *TODO:
 */
static void computeTransform(const Eigen::MatrixXf &points,
                             Eigen::Matrix3f &T) {
  Eigen::Vector2f mean = Eigen::Vector2f::Zero();
  Eigen::Vector2f variance = Eigen::Vector2f::Zero();
  for (Eigen::MatrixXf::Index i = 0; i < points.rows(); i++) {
    mean(i) += points.row(i).array().sum();
    variance(i) +=
        (points.row(i).array() * points.row(i).array()).array().sum();
  }
  for (Eigen::MatrixXf::Index i = 0; i < points.rows(); i++) {
    mean(i) /= points.cols();
    variance(i) = variance(i) / points.cols() - mean(i) * mean(i);
  }

  float xfactor = 1. / std::sqrt((float) variance[0]);
  float yfactor = 1. / std::sqrt((float) variance[1]);

  // variance很小时不做转换
  if (variance(0) < 1e-8) {
    xfactor = 1.0;
  }
  if (variance(1) < 1e-8) {
    yfactor = 1.0;
  }

  T << xfactor, 0, 1 - xfactor * mean(0), 0, yfactor, 1 - yfactor * mean(1), 0, 0, 1;
}

static void transformToPoints(const Eigen::MatrixXf &points,
                              const Eigen::Matrix3f &T,
                              Eigen::MatrixXf& transformed_points) {
  Eigen::MatrixXf ph = points.colwise().homogeneous();  //3xn
  Eigen::MatrixXf tph = T * ph;
  transformed_points = tph.colwise().hnormalized().topRows(2);
}

/** \brief 根据HZ p109的归一化步骤
 *
 */
static void normalizePoints(const Eigen::MatrixXf &points,
                            Eigen::MatrixXf &normalized_points,
                            Eigen::Matrix3f &T) {
  computeTransform(points, T);
  transformToPoints(points, T, normalized_points);
}

/** \brief
 *
 */
template<typename SolverArg,
         typename ErrorArg,
         typename ModelArg = Eigen::Matrix3f>
class Kernel {
 public:
  Kernel(const Eigen::MatrixXf &x1, const Eigen::MatrixXf &x2) : x1_(x1), x2_(x2) {}
  typedef SolverArg Solver;
  typedef ModelArg  Model;
  /// The minimal number of point required for the model estimation
  enum { MINIMUM_SAMPLES = Solver::MINIMUM_SAMPLES };
  /// The number of models that the minimal solver could return.
  enum { MAX_MODELS = Solver::MAX_MODELS };

  /// Extract required sample and fit model(s) to the sample
  void fit(const std::vector<size_t> *samples, std::vector<Model> *models) const {
    Eigen::MatrixXf x1 = extractColumns(x1_, samples);
    Eigen::MatrixXf x2 = extractColumns(x2_, samples);
    Solver::solve(x1, x2, models);
    std::cout << error((*samples)[0], (*models)[0]) << std::endl;
  }
  /// Return the error associated to the model and sample^nth point
  float error(size_t sample, const Model &model) const {
    return ErrorArg::error(model, x1_.col(sample), x2_.col(sample));
  }
  /// Number of putative point
  size_t numSamples() const {
    return x1_.cols();
  }
  /// Compute a model on sampled point
  static void solve(const Eigen::MatrixXf  &x1, const Eigen::MatrixXf  &x2, std::vector<Model> &models) {
    // By offering this, Kernel types can be passed to templates.
    Solver::solve(x1, x2, models);
  }
 public:
  const Eigen::MatrixXf & x1_, & x2_; // Left/Right corresponding point
};

/** \brief
 *
 */
template<typename SolverArg, typename ErrorArg,
    typename ModelArg = Eigen::Matrix3f>
class EssentialKernel : public Kernel<SolverArg, ErrorArg, ModelArg> {
 public:
  EssentialKernel(const Eigen::MatrixXf &x1, const Eigen::MatrixXf &x2,
                  const Eigen::Matrix3f &K1, const Eigen::Matrix3f &K2)
      : Kernel<SolverArg, ErrorArg, ModelArg>(x1, x2),
        K1_(K1),
        K2_(K2) {
  }

  void fit(const std::vector<size_t> *samples,
           std::vector<ModelArg> *models) const {
    Eigen::MatrixXf x1 = extractColumns(this->x1_, samples);
    Eigen::MatrixXf x2 = extractColumns(this->x2_, samples);

    assert(2 == x1.rows());
    assert(SolverArg::MINIMUM_SAMPLES <= x1.cols());
    assert(x1.rows() == x2.rows());
    assert(x1.cols() == x2.cols());

    // Normalize the data (image coords to camera coords).
    Eigen::Matrix3f K1Inverse = K1_.inverse();
    Eigen::Matrix3f K2Inverse = K2_.inverse();
    Eigen::MatrixXf x1_normalized, x2_normalized;
    transformToPoints(x1, K1Inverse, x1_normalized);
    transformToPoints(x2, K2Inverse, x2_normalized);
    SolverArg::solve(x1_normalized, x2_normalized, models);
  }

  float error(size_t sample, const ModelArg &model) const {
    //使用Fundamental
    Eigen::Matrix3f F = K2_.inverse().transpose() * model * K2_.inverse();
    return ErrorArg::error(F, this->x1_.col(sample), this->x2_.col(sample));
  }
 protected:
  Eigen::Matrix3f K1_, K2_;  // The two camera calibrated camera matrix
};

/*! \brief warp solver using normalized data fitting model.
 *
 */
template<typename SolverArg, typename UnnormalizerArg,
    typename ModelArg = Eigen::Matrix3f>
class NormalizedSolver {
 public:
  enum {
    MINIMUM_SAMPLES = SolverArg::MINIMUM_SAMPLES
  };
  enum {
    MAX_MODELS = SolverArg::MAX_MODELS
  };

  static void solve(const Eigen::MatrixXf &x1, const Eigen::MatrixXf &x2,
                    std::vector<ModelArg> *models) {
    assert(2 == x1.rows());
    assert(MINIMUM_SAMPLES <= x1.cols());
    assert(x1.rows() == x2.rows());
    assert(x1.cols() == x2.cols());

    // Normalize the data.
    Eigen::Matrix3f T1, T2;
    Eigen::MatrixXf x1_normalized, x2_normalized;
    normalizePoints(x1, x1_normalized, T1);
    normalizePoints(x2, x2_normalized, T2);

    SolverArg::solve(x1_normalized, x2_normalized, models);
    // Unormalize model from the computed conditioning.
    for (size_t i = 0; i < models->size(); ++i) {
      UnnormalizerArg::unnormalize(T1, T2, (*models)[i]);
    }
  }
};


struct AsymmetricHomographyError {
  static float error(const Eigen::MatrixXf &H, const Eigen::Vector2f &x1,
                     const Eigen::Vector2f &x2) {
    Eigen::Vector3f x1h(x1(0), x1(1), 1.0);
    Eigen::Vector3f x2h_est = H * x1h;
    Eigen::Vector2f x2_est = x2h_est.head<2>() / x2h_est[2];
    return (x2 - x2_est).squaredNorm();
  }
};

struct SymmetricHomographyError {
  static float error(const Eigen::MatrixXf &H, const Eigen::Vector2f &x1,
                     const Eigen::Vector2f &x2) {
    return AsymmetricHomographyError::error(H, x1, x2)
        + AsymmetricHomographyError::error(H.inverse(), x2, x1);
  }
};

// See page 288 equation (11.10) of HZ.
struct SymmetricEpipolarDistanceError {
  static double error(const Eigen::Matrix3f &F, const Eigen::Vector2f &x1,
                      const Eigen::Vector2f &x2) {
    Eigen::Vector3f xh(x1(0), x1(1), 1.0);
    Eigen::Vector3f yh(x2(0), x2(1), 1.0);

    Eigen::Vector3f F_x = F * xh;
    Eigen::Vector3f Ft_y = F.transpose() * yh;
    return pow(yh.dot(F_x), 2)
        * (1.0 / F_x.head<2>().squaredNorm()
            + 1.0 / Ft_y.head<2>().squaredNorm());
  }
};

struct SampsonError {
  static double error(const Eigen::Matrix3f &F, const Eigen::Vector2f &x1,
                      const Eigen::Vector2f &x2) {
    Eigen::Vector3f x(x1(0), x1(1), 1.0);
    Eigen::Vector3f y(x2(0), x2(1), 1.0);
    // See page 287 equation (11.9) of HZ.
    Eigen::Vector3f F_x = F * x;
    Eigen::Vector3f Ft_y = F.transpose() * y;
    return pow(y.dot(F_x), 2)
        / (F_x.head<2>().squaredNorm() + Ft_y.head<2>().squaredNorm());
  }
};

//参考HZ p109 Algo4.2
struct FourPointHomographySolver {
  enum {
    MINIMUM_SAMPLES = 4
  };
  enum {
    MAX_MODELS = 1
  };
  /**
   * Computes the homography that transforms x to y with the Direct Linear
   * Transform (DLT).
   * The estimated homography should approximately hold the condition y = H x.
   */
  static void solve(const Eigen::MatrixXf &x, const Eigen::MatrixXf &y,
                    std::vector<Eigen::Matrix3f> *Hs);

};

/**
 * Eight-point algorithm for solving for the essential matrix from normalized
 * image coordinates of point correspondences.
 * See page 294 in HZ Result 11.1.
 *
 */
struct EightPointEssentialSolver {
  enum {
    MINIMUM_SAMPLES = 8
  };
  enum {
    MAX_MODELS = 1
  };
  static void solve(const Eigen::MatrixXf &x, const Eigen::MatrixXf &y,
                    std::vector<Eigen::Matrix3f> *Es);
};


//struct FivePointEssentialSolver {
//  enum {
//    MINIMUM_SAMPLES = 5
//  };
//  enum {
//    MAX_MODELS = 10
//  };
//  static void solve(const Eigen::MatrixXf &x, const Eigen::MatrixXf &y,
//                    std::vector<Eigen::Matrix3f> &E);
//};

struct EightPointFundamentalSolver {
  enum {
    MINIMUM_SAMPLES = 8
  };
  enum {
    MAX_MODELS = 1
  };
  static void solve(const Eigen::MatrixXf &x1, const Eigen::MatrixXf &x2,
                    std::vector<Eigen::Matrix3f> *Fs);
};

template<typename TMat, typename TVec, int rank=-1>
bool nullspace(const TMat& A, TVec& x, bool rankCheck) {
  Eigen::JacobiSVD<TMat> svd(A, Eigen::ComputeFullV);
  typename Eigen::JacobiSVD<TMat>::SingularValuesType singular = svd.singularValues();
  //rank 1 check
  if (rankCheck) {
    if (svd.rank() != rank) {
      return false;
    }
  }

  x = svd.matrixV().col(A.cols() - 1);

  return true;
}

// HZ 12.2 pag.312
typedef Eigen::Matrix<float, 3, 4> MatrixP;
void triangulate(const MatrixP &P1, const Eigen::Vector2f &x1,
                 const MatrixP &P2, const Eigen::Vector2f &x2,
                 Eigen::Vector3f& X);

} // namespace carrotslam
#endif  /* CORE_MVG_H_ */
