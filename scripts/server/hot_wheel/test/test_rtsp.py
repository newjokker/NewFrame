# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import cv2
import os
import time


class RtspReceive():

    def __init__(self, rtsp, save_dir=None):
        self.rtsp = rtsp
        self.cap = cv2.VideoCapture(self.rtsp)
        self.save_dir = save_dir

    def get_img(self, save_name=False):
        ret, frame = self.cap.read()
        if save_name:
            each_save_path = os.path.join(self.save_dir, "{0}.jpg".format(save_name))
            cv2.imwrite(each_save_path, frame)
        else:
            return frame


if __name__ == "__main__":

    rtsp_path = "rtsp://admin:admin123@192.168.3.52:554/Streaming/Channels/101"
    save_dir = r"C:\Users\14271\Desktop\del\del"

    a = RtspReceive(rtsp_path, save_dir)

    for i in range(10):
        a.get_img(str(i))













