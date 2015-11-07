#ifndef GLVIEWER_H
#define GLVIEWER_H

#include <QGLViewer/qglviewer.h>

class Viewer : public QGLViewer
{
protected :
  virtual void init();
  virtual void draw();
  virtual QString helpString() const;
};


#endif // GLVIEWER_H

