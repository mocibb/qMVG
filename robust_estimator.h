/*!
 * Author: mocibb mocibb@163.com
 * Group:  CarrotSLAM https://github.com/mocibb/CarrotSLAM
 * Name:   robust_estimator.h
 * Date:   2015.10.28
 * Func:   implement robust estimation method.
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
#ifndef CORE_ROBUST_ESTIMATOR_H_
#define CORE_ROBUST_ESTIMATOR_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>
#include <random>

namespace carrotslam {
/*！
 * 平均采样num_samples个样本
 */
inline void uniformSample(size_t num_samples, size_t total_samples,
                          std::vector<size_t>* samples) {
  samples->clear();
  std::random_device rd;
  std::default_random_engine e1(rd());
  std::uniform_int_distribution<size_t> uniform_dist(0, total_samples-1);
  while (samples->size() < num_samples) {
    size_t sample = uniform_dist(e1);
    bool bFound = false;
    for (size_t j = 0; j < samples->size(); ++j) {
      bFound = ((*samples)[j] == sample);
      if (bFound) {  //the picked index already exist
        break;
      }
    }
    if (!bFound) {
      samples->push_back(sample);
    }
  }
}

/*！
 * 计算循环次数，循环次数为\f$ n \f$ ，\f$ n \f$ 次采样均采得外点的概率为outliers_probability
 * \f$(1-(inlier_ratio)^min_samples)^n=outliers_probability\f$
 */
inline size_t iterationsRequired(std::size_t min_samples,
                                 float outliers_probability,
                                 float inlier_ratio) {
  return static_cast<std::size_t>(std::log(outliers_probability)
      / std::log(1.0 - std::pow(inlier_ratio, static_cast<int>(min_samples))));
}




/*！
 * \brief The famous Random Sample Consensus algorithm (Fischler&Bolles 1981).
 *  Kernel 计算模型
 *  Scorer 评价模型
 */
template<typename Kernel, typename Scorer>
typename Kernel::Model RANSAC(const Kernel &kernel, const Scorer &scorer,
                              std::vector<size_t> *best_inliers = NULL,
                              float *best_score = NULL,
                              float outliers_probability = 1e-2,
                              size_t really_max_iterations = 4096) {
  assert(outliers_probability < 1.0);
  assert(outliers_probability > 0.0);
  size_t iteration = 0;
  const size_t min_samples = Kernel::MINIMUM_SAMPLES;
  const size_t total_samples = kernel.numSamples();
  //随着best_inlier_ratio调整
  size_t max_iterations = 100;

  size_t best_num_inliers = 0;
  float best_cost = std::numeric_limits<float>::infinity();
  float best_inlier_ratio = 0.0;
  typename Kernel::Model best_model;

  // 如果总样本数目不足报错
  if (total_samples < min_samples) {
    throw std::runtime_error("not enough sample!");
  }

  // In this robust estimator, the scorer always works on all the data points
  // at once. So precompute the list ahead of time [0,..,total_samples].
  std::vector<size_t> all_samples(total_samples);
  std::iota(all_samples.begin(), all_samples.end(), 0);

  std::vector<size_t> sample;
  for (iteration = 0;
      iteration < max_iterations && iteration < really_max_iterations;
      ++iteration) {
    //采样
    uniformSample(min_samples, total_samples, &sample);

    std::vector<typename Kernel::Model> models;
    kernel.fit(&sample, &models);

    // 评价模型
    for (size_t i = 0; i < models.size(); ++i) {
      std::vector<size_t> inliers;
      float cost = scorer.score(kernel, models[i], &all_samples, &inliers);

      if (cost < best_cost) {
        best_cost = cost;
        best_inlier_ratio = inliers.size() / float(total_samples);
        best_num_inliers = inliers.size();
        best_model = models[i];
        if (best_inliers) {
          best_inliers->swap(inliers);
        }

        if (best_inlier_ratio) {
          //重新根据best_inlier_ratio调整循环次数
          max_iterations = iterationsRequired(min_samples, outliers_probability,
                                              best_inlier_ratio);
          std::cout << "decrease iteration to "  << max_iterations << std::endl;
        }
      } // if (cost < best_cost) {
    }//for (size_t i = 0; i < models.size(); ++i) {
  }
  if (best_score)
    *best_score = best_cost;
  return best_model;
}


/// Templated Functor class to evaluate a given model over a set of samples.
template<typename Kernel>
class ScorerEvaluator {
 public:
  ScorerEvaluator(float threshold)
      : threshold_(threshold) {
  }

  template<typename T>
  float score(const Kernel &kernel, const typename Kernel::Model &model,
               const std::vector<T> *samples, std::vector<T> *inliers) const {
    float cost = 0.0;
    for (size_t j = 0; j < samples->size(); ++j) {
      float error = kernel.error((*samples)[j], model);
      if (error < threshold_) {
        cost += error;
        inliers->push_back((*samples)[j]);
      } else {
        cost += threshold_;
      }
    }
    return cost;
  }
 private:
  float threshold_;
};

} // namespace carrotslam
#endif  /* CORE_ROBUST_ESTIMATOR_H_ */
