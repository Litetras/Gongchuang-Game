from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
import time
import cv2
from PyQt5 import QtGui
from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QImage, QPixmap
from detect666shensai import rundetect
import serial as ser
se = ser.Serial("/dev/ttyTHS1", 9600, timeout=1)

tong = ['可回收物', '其他垃圾', '厨余垃圾', '有害垃圾']
detail = ['水瓶','易拉罐', '碎瓷片', '鹅卵石', '胡萝卜', '土豆', '电池', '药品']
#0水瓶，1易拉罐，2碎瓷片，3鹅卵石，4胡萝卜，5土豆，6电池，7药品
def data_recv(recv_data):
    head = b'\x53\x5A\x48\x59'
    head_ptr = 0
    head_find_flag = 0
    if recv_data:  # 判断是否有数据,根据接收数据类型修改
        for x in range(0, len(recv_data) - 8):  # 寻找帧头
            if recv_data[x:x+4] == head:
                head_ptr=x  # 记录帧头位置
                head_find_flag=1
                break
        if head_find_flag:  # 找到帧头
            cmd = recv_data[head_ptr + 5]  # 获取命令字
            if cmd == 0x01:
                print(recv_data[head_ptr + 6])

def data_send(cmd , data):#传入数据为bytes类型，
    head = b'\x59\x48\x5A\x53'
    send = head +cmd+data
    se.write(send)

class DetectThread(QThread):
    def __init__(self, display):
        QThread.__init__(self)
        self.display = display
    def run(self):
        self.display.Displaa()
class Display():
    def __init__(self, ui, mainWnd):
        self.ui = ui  # Assign ui object to ui attribute

        self.detectThread = DetectThread(self)

        self.startThreadTimer = QTimer()
        self.startThreadTimer.setSingleShot(True)
        self.startThreadTimer.timeout.connect(self.start_thread)
        self.startThreadTimer.start(1000)  # Delay 1 second

        self.detectThread.start()
        # 轮次垃圾桶初始化
        self.ke = 0
        self.hai = 0
        self.chu = 0
        self.other = 0
        self.count = 0

        # 表示第几个垃圾
        self.num = self.ke + self.hai + self.chu + self.other
        # 垃圾详细信息
        self.listView = ["本轮识别垃圾详情"]
        self.slm = QStringListModel()

        # 刷新界面数据
        self.refresh_num()

        # Load the video file and set the frame rate
        self.capfile = cv2.VideoCapture('1.avi')
        self.frameRate = self.capfile.get(cv2.CAP_PROP_FPS)

        # # 垃圾桶对应图片
        # self.ui.keui.setPixmap(QtGui.QPixmap('可回收物.png').scaled(64, 80))
        # self.ui.chuui.setPixmap(QtGui.QPixmap('厨余垃圾.png').scaled(64, 80))
        # self.ui.haiui.setPixmap(QtGui.QPixmap('有害垃圾.png').scaled(64, 80))
        # self.ui.otherui.setPixmap(QtGui.QPixmap('其他垃圾.png').scaled(64, 80))

        # 控制视频播放状态
        self.videoos = 0

        # 创建 QTimer 对象，用于定时刷新视频
        self.timer = QTimer()
        self.timer.timeout.connect(self.Displav)

    def refresh_num(self):
        self.ui.ke.setText(str(self.ke))
        self.ui.chu.setText(str(self.chu))
        self.ui.hai.setText(str(self.hai))
        self.ui.other.setText(str(self.other))
        self.slm.setStringList(self.listView)
        self.ui.listView.setModel(self.slm)

        # 重置本轮次垃圾投放信息

    def global_init(self):
        self.ke = 0
        self.hai = 0
        self.chu = 0
        self.other = 0
        self.count = 0
        self.kes = 0
        self.hais = 0
        self.chus = 0
        self.others = 0
        self.que = ''
        self.listView = ["本轮识别垃圾详情"]
        # 数据更新之后刷新界面
        self.refresh_num()

    def Displav(self):
        # Read the next frame from the video file
        success, frame = self.capfile.read()
        if success:
            # Convert the frame to a QImage and display it
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.disvideo.setPixmap(QPixmap.fromImage(img).scaled(721, 401))
        else:
            # Restart the video file from the beginning
            self.capfile.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.capfile.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.ui.disvideo.setPixmap(QPixmap.fromImage(img).scaled(721, 401))

    # 开始视频播放
    def start_video(self):
        if not self.timer.isActive():
            self.timer.start(int(self.frameRate))

    # tong = ['可回收物', '其他垃圾', '厨余垃圾', '有害垃圾']
    # detail = ['水瓶','易拉罐', '碎瓷片', '鹅卵石', '胡萝卜', '土豆', '电池', '药品']
    # # 0水瓶，1易拉罐，2碎瓷片，3鹅卵石，4胡萝卜，5土豆，6电池，7药品
    # 多次循环自动识别线程
    def Displaa(self):
        while True:
            print("检测开始")
            self.mode = rundetect()
            a_all=self.mode
            for a ,i in zip(a_all[0], range(0, len(a_all[1]), 4)):

                group = a_all[1][i:i + 4]

                if a == 0 or a == 1:
                    self.ke = self.ke + 1
                elif a == 2 or a == 3:
                    self.other = self.other + 1
                elif a == 4 or a == 5:
                    self.chu = self.chu + 1
                elif a == 6 or a == 7:
                    self.hai = self.hai + 1
                # 更新垃圾详细信息数据
                self.num = self.ke + self.hai + self.chu + self.other

                if a == 0 or a == 1:
                    add = "第" + str(self.num) + "个" + "   " + tong[a - 1] + "  " + str(self.ke) + '个' + "    ok!"
                    #串口输出种类
                elif a == 2 or a == 3:
                    add = "第" + str(self.num) + "个" + "   " + tong[a - 1] + "  " + str(self.other) + '个' + "    ok!"
                    # 串口输出种类
                elif a == 4 or a == 5:
                    add = "第" + str(self.num) + "个" + "   " + tong[a - 1] + "  " + str(self.chu) + '个' + "    ok!"
                    # 串口输出种类
                elif a == 6 or a == 7:
                    add = "第" + str(self.num) + "个" + "   " + tong[a - 1] + "  " + str(self.hai) + '个' + "    ok!"
                    # 串口输出种类
                print(add)
                ###串口输出坐标#######################################################
                print(group)
            # 更新垃圾详细信息界面显示
            self.listView.append(add)
            # 更新本轮次当前桶状态
            self.refresh_num()
            # 睡一觉，不看舵机干活
            time.sleep(12)
        # 等待一段时间后再继续检测

    def start_thread(self):
        self.detectThread.start()
