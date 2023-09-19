#!/usr/bin/env python
import sys
import uui #引入界面
from PyQt5.QtWidgets import QApplication, QMainWindow
from disa import Display #引入操作的类
import os
from PyQt5 import QtWidgets, QtGui, QtCore
import importlib
dis = importlib.import_module('dis')
from opencvvv import capture

if __name__ == '__main__':
    #os.system('fswebcam -r 1920x1080 --no-banner')
    #os.system('fswebcam -r 1920x1080 --no-banner')
    #os.system('fswebcam -r 1920x1080 --no-banner')
    stylesheet = """
              QMainWindow {  
                  background-image: url("2.png");  
                  background-repeat: no-repeat;
                  background-position: center;   
              }
          """
    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    mainWnd.setStyleSheet(stylesheet)
    ui = uui.Ui_MainWindow()
    ui.setupUi(mainWnd)
    ui.centralwidget.resize(1141, 700)
    ui.setupUi(mainWnd)
    display1 = Display(ui, mainWnd)
    display1.start_video()
    mainWnd.show()
    sys.exit(app.exec_())
