/*!
 * Author: mocibb mocibb@163.com
 * Group:  CarrotSLAM https://github.com/mocibb/CarrotSLAM
 * Name:   tum_dataset_reader.cpp
 * Date:   2015.09.30
 * Func:   tum dataset reader
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

#include "tum_dataset_reader.h"
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>


namespace carrotslam {
using namespace std;
using namespace boost::algorithm;
using namespace Sophus;

static bool readIntrinsic(const std::string &fileName, Eigen::Matrix3f &K)
{
  // Load the K matrix
  ifstream in;
  in.open( fileName.c_str(), ifstream::in);
  if(in.is_open())  {
    for (int j=0; j < 3; ++j)
      for (int i=0; i < 3; ++i)
        in >> K(j,i);
  }
  else  {
    std::cerr << std::endl
      << "Invalid input K.txt file" << std::endl;
    return false;
  }
  return true;
}

static void gtFromLine(std::string& line, GT& gt){
    gt.timestamp = -1;
    if (line[0] != '#') {
      vector<string> strs;
      boost::split(strs, line, is_space());

      gt.timestamp = std::atof(strs[0].c_str());
      Quaterniond qt(std::atof(strs[7].c_str()), std::atof(strs[4].c_str()),
                     std::atof(strs[5].c_str()), std::atof(strs[6].c_str()));
      Vector3d t;
      t << std::atof(strs[1].c_str()), std::atof(strs[2].c_str()), std::atof(
          strs[3].c_str());
      gt.pose = SE3d(qt, t);
    }
}

TUMDatasetReader::TUMDatasetReader(const string& path)
    : cnt_(0) {
  dataset_dir_ = path;

  string rgb_txt_path_ = dataset_dir_ + "/rgb.txt";
  string depth_txt_path_ = dataset_dir_ + "/depth.txt";
  string gt_txt_path_ = dataset_dir_ + "/groundtruth.txt";
  string intrinsc_txt_path_ = dataset_dir_ + "/K.txt";
  if (!boost::filesystem::exists(rgb_txt_path_)) {
    throw std::runtime_error("rgb.txt not exist!");
  }
  if (!boost::filesystem::exists(depth_txt_path_)) {
    throw std::runtime_error("depth.txt not exist!");
  }
  if (!boost::filesystem::exists(intrinsc_txt_path_)) {
    throw std::runtime_error("K.txt not exist!");
  }


  readIntrinsic(intrinsc_txt_path_ ,this->K_);

  ifstream rgb_txt_istream_(rgb_txt_path_);
  ifstream depth_txt_istream_(depth_txt_path_);
  ifstream gt_txt_istream_(gt_txt_path_);
  string line_;

  {
    while (getline(rgb_txt_istream_, line_)) {
      if (line_[0] != '#') {
        std::vector<std::string> strs;
        boost::split(strs, line_, is_space());
        rgb_dataset_.push_back(TUMDatasetImageLine(strs[0], strs[1]));
      }
    }

    while (getline(depth_txt_istream_, line_)) {
      if (line_[0] != '#') {
        std::vector<std::string> strs;
        boost::split(strs, line_, is_space());
        depth_dataset_.push_back(TUMDatasetImageLine(strs[0], strs[1]));
      }
    }

    GT gt;
    while (getline(gt_txt_istream_, line_)) {
      gtFromLine(line_, gt);
      if (gt.timestamp != -1) {
        gt_dataset_.push_back(gt);
      }
    }

    if (rgb_dataset_.size() != depth_dataset_.size()) {
      //LOG(ERROR) << "" << endl;
    }
  }

}

SE3d TUMDatasetReader::groundTruth(int frm) {
    int st = 0, end=gt_dataset_.size()-1;
    while(st+1<end) {
      int idx = st+round(0.5*(end-st));
      if (gt_dataset_[idx].timestamp < atof(rgb_dataset_[frm].timestamp.c_str())){
        st = idx;
      } else {
        end = idx;
      }
    }
    return gt_dataset_[st].pose;
}



}   // namespace carrotslam

