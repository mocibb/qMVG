#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTextCodec>
#include <QFileDialog>
#include <QPixmap>
#include <QTextCodec>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QMessageBox>
#include <QDebug>
#include <QGLViewer/qglviewer.h>
#include <string>
#include <sstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <Eigen/Core>
#include "tum_dataset_reader.h"
#include "mvg.h"
#include "robust_estimator.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace carrotslam;

namespace Ui {
class MainWindow;
}

enum ROBUST_ESTIMATOR{
    RANSAC,
    LMEDS,
    ACRANSAC
};

class MainWindow : public QMainWindow
{
    typedef Kernel<FourPointHomographySolver, SymmetricHomographyError> Hkernel;
    typedef Kernel<NormalizedSolver<EightPointFundamentalSolver, UnnormalizerE>, SampsonError> Fkernel;
    typedef EssentialKernel<NormalizedSolver<EightPointFundamentalSolver, UnnormalizerE>, SampsonError> Ekernel;

    Q_OBJECT
private slots:
    void selectTumDataset();
    void exit();
    void loadLeftImage();
    void loadRightImage();
    void computeMatches();
    void computeEssential();

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:    
    void loadImage();
    void writeOut();
    void findMatches();
    void showMatches();
    void loadDataset();
    void computeEssentialStatistics(const Ekernel& kernel, const Matrix3f &E);
    void clear();
    /**
     * 返回从右图坐标系到左图坐标系的变换
     * left_pose_是左图世界坐标系下坐标
     * right_pose_是右图世界坐标系下坐标
     */
    Sophus::SE3f pose() {
      Sophus::SE3f pose_ =  right_pose_.inverse() * left_pose_;
      return pose_;
    }

    ostream& info() {
        return info_;
    }

    Ui::MainWindow *ui;
    QTextCodec* codec_;
    TUMDatasetReader* tum_reader_ = nullptr;
    Mat left_img_;
    Mat right_img_;
    Mat merged_img_;
    string img_dir_;
    Ptr<FeatureDetector> detector_;
    Ptr<DescriptorExtractor> descriptor_;
    std::vector<cv::DMatch> matches_;
    std::vector<cv::KeyPoint> left_features_;
    std::vector<cv::KeyPoint> right_features_;
    cv::Mat left_desp_;
    cv::Mat right_desp_;
    std::vector<size_t> inliers_;
    stringstream info_;
    Sophus::SE3f left_pose_;
    Sophus::SE3f right_pose_;
    IOFormat mfmt_;
};

#endif // MAINWINDOW_H
