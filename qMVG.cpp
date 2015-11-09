#include "qMVG.h"
#include "ui_qMVG.h"
#include "mvg_motion.h"
#include <Eigen/Geometry>

using namespace carrotslam;
using namespace Eigen;
using namespace Sophus;


QImage* Mat2QImage(cv::Mat const& src)
{
     cv::Mat temp; // make the same cv::Mat
     cvtColor(src, temp,CV_BGR2RGB); // cvtColor Makes a copt, that what i need
     QImage* dest = new QImage((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
     dest->bits(); // enforce deep copy, see documentation
     // of QImage::QImage ( const uchar * data, int width, int height, Format format )
     return dest;
}

Mat QImage2Mat(QImage const& src)
{
     cv::Mat tmp(src.height(),src.width(),CV_8UC3,(uchar*)src.bits(),src.bytesPerLine());
     cv::Mat result; // deep copy just in case (my lack of knowledge with open cv)
     cvtColor(tmp, result,CV_BGR2RGB);
     return result;
}

QImage putImage(const Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        // Set the color table (used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    if(mat.type()==CV_8UC3)
    {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    else
    {
        //qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

void setQLabel(QLabel* lbl, const Mat& image, Mat& lblImage) {
    if (image.size().area() < 1) {
        return;
    }
    float w = lbl->width();
    float h = lbl->height();
    float radio = 1.f;
    radio = w>image.cols?radio:w/image.cols;
    radio = h>image.rows?radio:std::min(h/image.rows,radio);

    cv::resize(image, lblImage, Size(image.cols*radio,image.rows*radio));
    QImage img = putImage(lblImage);

    lbl->setPixmap(QPixmap::fromImage(img));
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->actionOpen_Dataset, SIGNAL(triggered()), this, SLOT(selectTumDataset()));
    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(exit()));
    connect(ui->loadLImg, SIGNAL(released()), this, SLOT(loadLeftImage()));
    connect(ui->loadRImg, SIGNAL(released()), this, SLOT(loadRightImage()));
    connect(ui->vFeature, SIGNAL(released()), this, SLOT(computeMatches()));
    connect(ui->vEssential, SIGNAL(released()), this, SLOT(computeEssential()));


    codec_ = QTextCodec::codecForName("UTF-8");

    cv::initModule_nonfree();
    detector_ = FeatureDetector::create("SURF");
    descriptor_ = DescriptorExtractor::create("SURF");

    tum_reader_ = new TUMDatasetReader("/home/mocibb/dataset/tum/rgbd_dataset_freiburg1_xyz");
    loadDataset();
}

void MainWindow::writeOut() {
    ui->logMsgBox->append(QString::fromStdString(info_.str()));

    info_.str( std::string() );
    info_.clear();
}


void MainWindow::loadDataset() {
    QStandardItemModel* model = new QStandardItemModel(this);

    int i=0;
    for(auto& it : tum_reader_->rgb_dataset_) {
        QString fileName = codec_->toUnicode((tum_reader_->dataset_dir_+"/"+it.filename).c_str());
        QFileInfo fi(fileName);
        img_dir_ = codec_->fromUnicode(fi.dir().absolutePath()).toStdString();
        QStandardItem *item = new QStandardItem(fi.fileName());
        if(i++ % 2 == 1)
        {
            QLinearGradient linearGrad(QPointF(0, 0), QPointF(200, 200));
            linearGrad.setColorAt(0, QColor("#ADD8E6"));
            linearGrad.setColorAt(1, Qt::white);
            QBrush brush(linearGrad);
            item->setBackground(brush);
        }
        model->appendRow(item);
    }
    this->ui->ds_list->setModel(model);
    this->ui->ds_list->setEditTriggers(QAbstractItemView::NoEditTriggers);
}

void MainWindow::selectTumDataset() {
    QString dirName = QFileDialog::getExistingDirectory(this, QStringLiteral("select tum dataset folder"), "");

    if (dirName == "") return;

    if (tum_reader_ != nullptr) {
        delete tum_reader_;
    }
    try {
        tum_reader_ = new TUMDatasetReader(dirName.toStdString());
    } catch(...) {
        QMessageBox::critical(this, "Error",
                            QStringLiteral("cannot open tum dataset!"));
        return;
    }

    loadDataset();
}

void MainWindow::exit() {
    QApplication::quit();
}

void MainWindow::loadLeftImage() {
    QStandardItemModel* smodel = static_cast<QStandardItemModel*>(ui->ds_list->model());
    QModelIndexList idx = ui->ds_list->selectionModel()->selectedIndexes();
    if (idx.size() == 1) {
        left_img_ = imread(img_dir_+"/"+smodel->itemFromIndex(idx[0])->text().toStdString());
        if (tum_reader_ != nullptr) {
            left_pose_ = tum_reader_->groundTruth(idx[0].row()).cast<float>();
        }
        loadImage();
    }
}

void MainWindow::loadRightImage(){
    QStandardItemModel* smodel = static_cast<QStandardItemModel*>(ui->ds_list->model());
    QModelIndexList idx = ui->ds_list->selectionModel()->selectedIndexes();
    if (idx.size() == 1) {
        right_img_ = imread(img_dir_+"/"+smodel->itemFromIndex(idx[0])->text().toStdString());
        if (tum_reader_ != nullptr) {
            right_pose_ = tum_reader_->groundTruth(idx[0].row()).cast<float>();
        }
        loadImage();
    }
}

void MainWindow::loadImage(){
    Size sz = left_img_.size();
    if (sz == Size()) {
        sz = right_img_.size();
    }

    if (!merged_img_.data) {
        if (left_img_.size() != Size()) {
            merged_img_.create(left_img_.rows, 2*left_img_.cols, left_img_.type());
        } else {
            merged_img_.create(right_img_.rows, 2*right_img_.cols, right_img_.type());
        }
    }
    if (left_img_.data) {
        left_img_.copyTo(Mat(merged_img_, cv::Rect(0,0,sz.width,sz.height)));
    }
    if (right_img_.data) {
        right_img_.copyTo(Mat(merged_img_, cv::Rect(sz.width,0,sz.width,sz.height)));
    }

    Mat lblImage;
    setQLabel(ui->imageLbl, merged_img_, lblImage);

}

void MainWindow::findMatches() {
    left_features_.clear();
    right_features_.clear();

    detector_->detect(left_img_, left_features_);
    detector_->detect(right_img_, right_features_);

    descriptor_->compute(left_img_, left_features_, left_desp_);
    descriptor_->compute(right_img_, right_features_, right_desp_);

    // 调用opencv::FlannBasedMatcher进行匹配
    FlannBasedMatcher matcher;
    std::vector<cv::DMatch> all_match;
    matcher.match( left_desp_, right_desp_, all_match );

    matches_.clear();
    auto min_ele = std::min_element( all_match.begin(), all_match.end(),
            [] (const cv::DMatch& m1, const cv::DMatch& m2 )
            { return m1.distance<m2.distance; });
    auto isGoodMatch = [min_ele, this] (const cv::DMatch& m)
        {return m.distance < min_ele->distance * this->ui->minDistRate->value(); } ;
    auto iter = all_match.begin();
    while ( true )
    {
        iter = find_if( iter, all_match.end(), isGoodMatch );
        if ( iter == all_match.end() )
            break;
        matches_.push_back(*iter);
        iter++;
    }

    info() << "threshold value:" << this->ui->minDistRate->value() << " found matched point: "  << matches_.size();
    writeOut();
}

static void drawKeypoint(Mat& img, const KeyPoint& kp, const Point2f shift=Point2f()) {
    cv::Point center = kp.pt + shift;
    int radius = 1 << kp.octave;
    Scalar color = Scalar(255, 0, 0);
    circle(img, center, radius, color, 1, CV_AA);

    // default value of angle is -1
    if (kp.angle != -1) {
        float srcAngleRad = kp.angle * (float) CV_PI / 180.f;
        cv::Point orient(cvRound(cos(srcAngleRad) * radius),
                         cvRound(sin(srcAngleRad) * radius));
        line(img, center, center + orient, color, 1, CV_AA);
    }
}


void MainWindow::showMatches() {
    Mat img = merged_img_.clone();
    Mat lblImage;

    sort(inliers_.begin(), inliers_.end());
    Scalar outlier(0,0,255), inlier(0, 255, 0);
    size_t j = 0;
    for (size_t i = 0; i < matches_.size(); i++) {
        if (ui->showLeft->isChecked()) {
            drawKeypoint(img, left_features_[matches_[i].queryIdx]);
        }
        if (ui->showRight->isChecked()) {
            drawKeypoint(img, right_features_[matches_[i].trainIdx], Point2f(merged_img_.cols*0.5, 0));
        }

        if (ui->showCrrsp->isChecked()) {
            if (inliers_.size()==0 || j>inliers_.size() || i!=inliers_[j]) {
                cv::line(img, left_features_[matches_[i].queryIdx].pt,
                         Point2f(right_features_[matches_[i].trainIdx].pt.x+merged_img_.cols*0.5,
                                 right_features_[matches_[i].trainIdx].pt.y),
                         outlier, 1);
            } else if (j<inliers_.size() && i==inliers_[j]) {
                cv::line(img, left_features_[matches_[i].queryIdx].pt,
                         Point2f(right_features_[matches_[i].trainIdx].pt.x+merged_img_.cols*0.5,
                                 right_features_[matches_[i].trainIdx].pt.y),
                         inlier, 1);
                j++;
            }
        }
    }
    setQLabel(ui->imageLbl, img, lblImage);
}

void MainWindow::computeMatches() {

    if (left_img_.data && right_img_.data) {
        clear();

        findMatches();

        showMatches();

    }
}

void MainWindow::computeEssentialStatistics(const Ekernel& kernel, const Matrix3f &E) {
    float thres = this->ui->ransacThres->value() / 100.f;
    //内点数
    int total_samples = kernel.numSamples();
    std::vector<size_t> all_samples(total_samples);
    std::vector<size_t> inliers;
    std::iota(all_samples.begin(), all_samples.end(), 0);
    ScorerEvaluator<Ekernel> scorer(thres);
    scorer.score(kernel, E, &all_samples, &inliers);
    //分解E
    Matrix3f R;
    Vector3f t;

    decomposeEssential(E, tum_reader_->K_, &(kernel.x1_), tum_reader_->K_, &(kernel.x2_), R, t);
    //投影误差
    vector<float> errors;
    errors.reserve(inliers.size());
    MatrixP p1;
    MatrixP p2;
    p1 = MatrixP::Zero();
    p1.block<3, 3>(0, 0) = Eigen::Vector3f::Ones().asDiagonal();
    p1 = tum_reader_->K_ * p1;
    p2.block<3, 3>(0, 0) = R;
    p2.block<3, 1>(0, 3) = t;
    p2 = tum_reader_->K_ * p2;

    for (size_t j=0; j<inliers.size(); j++){
        Vector3f X;
        triangulate(p1, kernel.x1_.col(inliers[j]), p2, kernel.x1_.col(inliers[j]), X);

        errors[j] = (p1 * X.homogeneous()).colwise().hnormalized().squaredNorm() +
                (p2 * X.homogeneous()).colwise().hnormalized().squaredNorm() - 2;
    }

    sort(errors.begin(), errors.end());
    float total_errors = std::accumulate(errors.begin(), errors.end(), 0);
    info() << "inliers: " << inliers.size() << endl;
    info() << "min_residual: " << errors[0] << endl;
    info() << "max_residual: " << errors[errors.size()-1] << endl;
    info() << "mean_residual: " << total_errors / errors.size() << endl;
}

void MainWindow::computeEssential() {
    if (left_img_.data && right_img_.data) {

        clear();

        findMatches();

        MatrixXf x1(2, matches_.size()), x2(2, matches_.size());
        int j = 0;
        for (size_t i = 0; i < matches_.size(); i++) {
            x1.col(j) << left_features_[matches_[i].queryIdx].pt.x,
                    left_features_[matches_[i].queryIdx].pt.y;
            x2.col(j) << right_features_[matches_[i].trainIdx].pt.x,
                right_features_[matches_[i].trainIdx].pt.y;
            j++;
        }

        float thres = this->ui->ransacThres->value() / 100.f;
        float best_score;
        Ekernel ekernel(x1, x2, tum_reader_->K_, tum_reader_->K_);

        Matrix3f E = carrotslam::RANSAC(ekernel, ScorerEvaluator<Ekernel>(thres), &inliers_, &best_score);
        //Matrix3f E = carrotslam::RANSAC(ekernel, ScorerEvaluator<Ekernel>(thres), &inliers_);
        info() << "estimatie essential matrix using threshold value:" << thres << endl;
        info() << "score: " << best_score
               << "\tE: " << E.row(0) << "; " << E.row(1) << "; " << E.row(2) << endl;

        showMatches();

        computeEssentialStatistics(ekernel, E);

        cout << pose().matrix() << endl;
        if (pose().translation() != Vector3f::Zero()) {
            info() << "ground truth" << endl;
            Matrix3f gtE;

            essentialFromMotion(gtE, pose().rotationMatrix(), pose().translation());
            computeEssentialStatistics(ekernel, gtE);
        }

        writeOut();
    }
}

void MainWindow::clear() {
    matches_.clear();
    left_features_.clear();
    right_features_.clear();
    left_desp_ = Mat();
    right_desp_ = Mat();
    inliers_.clear();
    left_pose_ = SE3f();
    right_pose_ = SE3f();
}

MainWindow::~MainWindow()
{
    delete ui;
}
