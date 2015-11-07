#-------------------------------------------------
#
# Project created by QtCreator 2015-11-02T20:32:25
#
#-------------------------------------------------

QT       += core gui opengl xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = qMVG
TEMPLATE = app

CONFIG += c++11

SOURCES += main.cpp\
    glviewer.cpp \
    mvg.cpp \
    tum_dataset_reader.cpp \
    qMVG.cpp

HEADERS  += \
    glviewer.h \
    mvg.h \
    robust_estimator.h \
    tum_dataset_reader.h \
    qMVG.h \
    mvg_motion.h

FORMS    += \
    qMVG.ui

INCLUDEPATH += /usr/include/eigen3 /home/mocibb/works/Sophus

LIBS += -L/usr/local/lib -lQGLViewer -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_flann -lopencv_nonfree -lboost_filesystem -lboost_system
