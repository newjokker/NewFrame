# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
import threading
import platform
from ..utils.FileOperationUtil import FileOperationUtil
from ..utils.StrUtil import StrUtil


class VideoUtilCV(object):

    def __init__(self, vedio_path):
        self.__vc = cv2.VideoCapture(vedio_path)

    def get_frame_img_in_assign_time(self, save_dir, hours=0, minutes=0, seconds=0, assign_frame_num=10, frame_rate=24):
        """获取指定时间范围内的指定帧，可以指定获取的帧的数目"""
        frame_start_num = (hours * 3600 + minutes * 60 + seconds) * frame_rate - int(assign_frame_num/2)
        frame_start_num = 0 if frame_start_num < 0 else frame_start_num
        self.__vc.set(cv2.CAP_PROP_POS_FRAMES, frame_start_num)  # 指定帧

        index = 0
        while True:
            read_res = self.__vc.read()  # 读取一帧影像
            cv2.imwrite(os.path.join(save_dir, '{0}.jpg'.format(index)), read_res[1])  # 将数据写到本地
            index += 1
            if index >= assign_frame_num:
                break

    @staticmethod
    def get_img_from_vedio(vedio_path, save_dir, sep=30, start_index=0):
        """将图像分出每一个帧"""
        vedio_name = os.path.splitext(os.path.split(vedio_path)[1])[0]
        i, index = 0, start_index
        while True:
            i += sep
            index += 1
            vc = cv2.VideoCapture(vedio_path)
            vc.set(cv2.CAP_PROP_POS_FRAMES, i)
            img = vc.read()

            if not img[0]:
                break

            try:
                each_save_path = os.path.join(save_dir, '{0}_{1}.jpg'.format(vedio_name, str(index).rjust(3, '0')))
                if StrUtil.contain_zh(each_save_path):
                    raise ValueError("save path contain chinese : {0}".format(each_save_path))
                cv2.imwrite(each_save_path, img[1])  # 将数据写到本地
            except Exception as e:
                print("error : {0} {1}".format(e, img))

    @staticmethod
    def write_vedio(img_path_list, out_path, assign_fps=20, assign_size=None):
        """图像拼接为视频"""
        fourcc = VideoWriter_fourcc(*"MJPG")  #支持jpg
        if not assign_size:
            img_shape = cv2.imread(img_path_list[0]).shape
            assign_size_temp = (img_shape[1], img_shape[0])
        else:
            assign_size_temp = assign_size
        videoWriter = cv2.VideoWriter(out_path, fourcc, assign_fps, assign_size_temp)

        # 写入每一张图像
        len_imgs = len(img_path_list)
        for index, each_img_path in enumerate(img_path_list):
            print("{0} / {1}".format(index, len_imgs))
            frame = cv2.imread(each_img_path)
            if assign_size:
                frame = cv2.resize(frame, assign_size)
            videoWriter.write(frame)
        videoWriter.release()

    @staticmethod
    def get_img_from_rstp():
        """从 rstp 视屏流拉视频"""
        # TODO 将内容机型一下整理
        vid = cv2.VideoCapture(r"rtsp://admin:tx666999@192.168.3.64:554/h264/ch1/main/av_stream")
        ret, img_ori = vid.read()               # 获取视频画面信息
        video_frame_cnt = int(vid.get(7))
        video_width = int(vid.get(3))
        video_height = int(vid.get(4))
        video_fps = int(vid.get(5))


class CompressVideo(object):
    """视频压缩，其实就是格式转换，一般 avi 转为 mp4"""

    def __init__(self, dir_path, input_name, out_name=""):
        self.filePath = dir_path
        self.inputName = input_name
        self.outName = out_name
        # fixme 要是 system 只是为了拼接路径使用，为什么不使用两个系统都能识别的路径样式
        self.system_ = platform.platform().split("-",1)[0]
        if  self.system_ ==  "Windows":
            self.filePath = (self.filePath + "\\") if self.filePath.rsplit("\\",1)[-1] else self.filePath
        elif self.system_ == "Linux":
            self.filePath = (self.filePath + "/") if self.filePath.rsplit("/",1)[-1] else self.filePath
        self.fileInputPath = self.filePath + input_name
        self.fileOutPath = self.filePath + out_name

    @property
    def is_video(self):
        videoSuffixSet = {"WMV","ASF","ASX","RM","RMVB","MP4","3GP","MOV","M4V","AVI","DAT","MKV","FIV","VOB"}
        suffix = self.fileInputPath.rsplit(".",1)[-1].upper()
        if suffix in videoSuffixSet:
            return True
        else:
            return False

    def SaveVideo(self):
        fpsize = os.path.getsize(self.fileInputPath) / 1024
        if fpsize >= 150.0: #大于150KB的视频需要压缩
            if self.outName:
                compress = "ffmpeg -i {} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf 23 -acodec aac -b:a 32k -strict -5 {}".format(self.fileInputPath,self.fileOutPath)
                isRun = os.system(compress)
            else:
                compress = "ffmpeg -i {} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf 23 -acodec aac -b:a 32k -strict -5 {}".format(self.fileInputPath, self.fileInputPath)
                isRun = os.system(compress)
            if isRun != 0:
                return (isRun, "没有安装ffmpeg")
            return True
        else:
            return True

    def Compress_Video(self):
        # 异步保存打开下面的代码，注释同步保存的代码
        thr = threading.Thread(target=self.SaveVideo)
        thr.start()
        #下面为同步代码
        # fpsize = os.path.getsize(self.fileInputPath) / 1024
        # if fpsize >= 150.0:  # 大于150KB的视频需要压缩
        #     compress = "ffmpeg -i {} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf 23 -acodec aac -b:a 32k -strict -5 {}".format(
        #         self.fileInputPath, self.fileOutPath)
        #     isRun = os.system(compress)
        #     if isRun != 0:
        #         return (isRun, "没有安装ffmpeg")
        #     return True
        # else:
        #     return True

    # ------------------------------------------------------

    @staticmethod
    def demo():
        """例子"""
        filePath = r"C:\Users\14271\Desktop\vedio_test"             # 视频保存的文件夹
        inputName = r"test_0.avi"                                   # 视频文件名
        outName = r"test_1.mp4"                                     # 转换格式后的视频文件名
        savevideo = CompressVideo(filePath, inputName, outName)
        print(savevideo.Compress_Video())



if __name__ == '__main__':


    # # for each_vedio_path in FileOperationUtil.re_all_file(r"C:\Users\14271\Desktop\test\vedio"):
    # for each_vedio_path in FileOperationUtil.re_all_file(r"C:\Users\14271\Desktop\test_vedio"):
    #     dir_name, file_name = os.path.split(each_vedio_path)
    #     file_name, suffix = os.path.splitext(file_name)
    #     save_folder = os.path.join(dir_name, file_name)
    #
    #     if os.path.exists(save_folder):
    #         shutil.rmtree(save_folder)
    #         os.makedirs(save_folder)
    #     else:
    #         os.makedirs(save_folder)
    #
    #     VideoUtilCV.get_img_from_vedio(each_vedio_path, save_folder, sep=1, start_index=0)


    out_path = r"C:\Users\14271\Desktop\test_vedio_all_005.avi"
    img_folder = r"C:\Users\14271\Desktop\all_005"
    # a.write_vedio(FileOperationUtil.re_all_file(save_folder), out_path, assign_fps=5, assign_size=(640, 480))

    file_list = FileOperationUtil.re_all_file(img_folder)

    # file_list = file_list[::-1]

    VideoUtilCV.write_vedio(file_list, out_path, assign_fps=15)




