# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 参考: https://www.testwo.com/blog/7270

import ftplib
import os
import logging


class FTPUtil(object):

    def __init__(self):
        self.__ftp = None
        self.__user_name = None
        self.__password = None
        self.__host = None
        self.__ip = None
        self.__port = None

    def login(self, user_name, password, host='', ip=None, port=None):
        """login"""
        self.__user_name = user_name
        self.__password = password
        self.__host = host
        self.__ip = ip
        self.__port = port

        ftp = ftplib.FTP(host)
        if ip and port:
            ftp.connect(ip, port)
        else:
            ftp.connect(host, port)

        ftp.login(user_name, password)
        self.__ftp = ftp

    def ftp_download(self, file_ftp_path, file_local_path, bufsize=1024*10):
        """download file as binary mode"""
        # check if file is exist
        file_dir = os.path.dirname(file_local_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(file_local_path, 'wb') as ftp_file:
            self.__ftp.retrbinary('RETR %s' % file_ftp_path, ftp_file.write, bufsize)

    def listdir(self, scan_folder, return_complete_path=False):
        """scan folder return child folder and file"""
        file_paths = []
        # get file
        for each_file in self.__ftp.nlst(scan_folder):
            file_paths.append(each_file)
        # if full path is needed
        if return_complete_path:
            return map(lambda x: os.path.join(scan_folder, x), file_paths)
        else:
            return file_paths

    def clone_dir(self, sourse_dir, target_dir, assign_endwith=None, print_detail=False):
        """clone a folder to assign path"""
        # scane file
        all_file_path = self.listdir(sourse_dir)
        file_count = len(all_file_path)
        for file_index, each_file in enumerate(all_file_path):
            # suffix checking
            if assign_endwith:
                if not each_file.endswith(tuple(assign_endwith)):
                    continue

            # source path
            source_file = os.path.join(sourse_dir, each_file)
            # local path
            target_file = os.path.join(target_dir, each_file)
            # load
            self.ftp_download(source_file, target_file)
            #
            logging.info("A total of {0} file，now scan {1} ".format(len(all_file_path), file_index))
            if print_detail:
                print("{0} / {1} --> {2} ".format(file_index+1, file_count, target_file))

    def close(self):
        """close ftp"""
        # self.ftp.clode()
        self.__ftp.quit()

    # --------------------------------------- need repair --------------------------------------------------------------

    def cteate_remote_folder(self, folder_path):
        """create a remote directory"""
        self.__ftp.mkd(folder_path)

    def re_login(self):
        """if connect is break, connect again"""
        self.login(self.__user_name, self.__password, self.__host, self.__ip, self.__port)

    def is_connect(self):
        """if the ftp is in connect"""
        pass

    def del_remote_file(self, file_path):
        """delete remote file"""
        self.__ftp.delete(file_path)

    def del_remote_folder(self, folder_path):
        """delete remote folder"""
        self.__ftp.rmd(folder_path)

    def rename_remote_file_aor_folder(self, old_name, new_name):
        """rename remote file or folder"""
        self.__ftp.rename(old_name, new_name)

    def up_load_file(self, file_path):
        """up load file"""
        fp = open(file_path, 'rb')
        cmd_str = 'STOR {0}'.format(file_path)
        self.__ftp.storbinary(cmd_str, fp)


if __name__ == '__main__':

    Host = 'rsapp.nsmc.org.cn'
    # Ip = 'ftp://192.168.1.236'
    userName = 'suanfa'
    passWord = 'suanfa123'
    a = FTPUtil()
    a.login(userName, passWord, port=2221, host=Host)
    a.clone_dir(r'/fire/auxdata/yanbian', r'C:\Users\Administrator\Desktop\new', print_detail=True)
