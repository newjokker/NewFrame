# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import py_compile
import os, sys
from Crypto.Cipher import AES
import random, struct
import shutil
import argparse
import sys


"""
* 使用环境 ENV 加密文件夹 A，B 不加密文件夹 A 中的 C文件夹 和 d 文件，加密文件 a ,b，删除加密的 py 文件，对要加密的文件夹进行拷贝
python3 setup.py -end A,B -exd A/C -exf d -enf a,b -env ENV -c -o 
"""

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-end", "--entrypt_dir", type=str)      # 需要加密的文件夹
    ap.add_argument("-exd", "--exclude_dir", type=str)      # 不需要加密的文件夹
    ap.add_argument("-enf", "--entrypt_file", type=str)     # 需要加密的文件
    ap.add_argument("-exf", "--exclude_file", type=str)     # 不需要加密的文件
    ap.add_argument("-env", "--env_path", type=str)         # 环境
    ap.add_argument("-enm", "--encrypt_mode", type=str)     # 环境
    ap.add_argument("-c", "--clear", action="store_true")   # 删除 py 文件
    ap.add_argument("-o", "--old_dir", action="store_true") # 是否复制文件生成 _old 文件夹
    args = ap.parse_args()
    return args


class EncryptPy(object):

    def __init__(self):
        self.entrypt_mode = "so"        # 加密的方式 so | enc 代表生成文件的后缀
        self.is_clear = False
        self.is_zip = False
        self.is_copy = False            # 是否要复制一份文件
        self.entrypt_dir_list = []      # 要编译的文件夹
        self.entrypt_file_list = []     # 要编译的文件
        self.exclude_dir_list = []      # 要排除的文件夹
        self.exclude_file_list = []     # 要排除的文件
        self.env_path = r"/usr/include/python3.5/"

    @staticmethod
    def re_all_file(file_path, func=None):
        """返回文件夹路径下的所有文件路径（搜索文件夹中的文件夹）"""
        if not os.path.isdir(file_path):
            raise EOFError

        result = []
        for i, j, k in os.walk(file_path):
            for each in k:
                abs_path = i + os.sep + each
                if func is None:  # is 判断是不是指向同一个东西
                    result.append(abs_path)
                else:
                    if func(abs_path):
                        result.append(os.path.join(i, each))
        return result

    def do_init(self):
        """初始化"""
        if not os.path.exists(self.env_path):
            raise ValueError("env error !!!")

    def get_all_file_need_entrype(self):
        """找到所有需要编译的文件"""
        # find all file can be entrype
        file_set = set(self.entrypt_file_list)
        for each_file_dir in self.entrypt_dir_list:
            each_file_list = self.re_all_file(each_file_dir, lambda x: str(x).endswith(('.py', '.pyc', '.so')))
            file_set.update(each_file_list)
        # find all file should be exclude
        exclude_file_set = set(self.exclude_file_list)
        for each_dir in self.exclude_dir_list:
            each_file_list = self.re_all_file(each_dir, lambda x: str(x).endswith(('.py', '.pyc')))
            exclude_file_set.update(set(each_file_list))
        # remove file should be exclude
        self.entrypt_file_list = file_set - exclude_file_set

    def copy_dir(self):
        """复制文件，文件后面加 _old"""

        if not self.is_copy:
            return

        for each_dir in self.entrypt_dir_list:
            print("copy dir : {0}".format(each_dir))
            if not os.path.isdir(each_dir):
                raise ValueError("need dir ： {0}".format(each_dir))
            else:
                new_dir = each_dir.rstrip("\//") + "_old"
                print(each_dir)
                print(new_dir)
                if os.path.exists(new_dir):
                    print("_old dir is exists : {0}".format(new_dir))
                else:
                    shutil.copytree(each_dir, new_dir)

    def encrypt_so(self, dir_pref):
        """编译 python3, 生成 so 文件"""
        cmd_str_001 = 'cython -2 {1}.py; gcc -c -fPIC -I {0} {1}.c -o {1}.o'.format(self.env_path, dir_pref)
        os.system(cmd_str_001)
        cmd_str_002 = 'gcc -shared {0}.o -o {0}.so'.format(dir_pref)
        os.system(cmd_str_002)
        os.system('rm -f {0}.c {0}.o'.format(dir_pref))

    @staticmethod
    def encrypt_enc(in_filename, key=None, out_filename=None, chunksize=64 * 1024):
        """
        使用AES（CBC模式）加密文件给定的密钥。
        :param key: 加密密钥-必须是16、24或32字节长。长按键更安全。
        :param in_filename: 输入的文件的名称
        :param out_filename: 如果为 None，将使用“<in_filename>.enc”。
        :param chunksize: 设置函数用于读取和加密文件。大块一些文件和机器的大小可能更快。块大小必须可被16整除。
        :return: None
        """
        if not out_filename:
            out_filename = in_filename + '.enc'
        salt = 'txkj2019'  # 盐值
        if key is None:
            key = "{: <32}".format(salt).encode("utf-8")
        iv = b'0000000000000000'
        encryptor = AES.new(key, AES.MODE_CBC, iv)
        filesize = os.path.getsize(in_filename)

        with open(in_filename, 'rb') as infile:
            with open(out_filename, 'wb') as outfile:
                outfile.write(struct.pack('<Q', filesize))
                outfile.write(iv)
                while True:
                    chunk = infile.read(chunksize)
                    if len(chunk) == 0:
                        break
                    elif len(chunk) % 16 != 0:
                        chunk += (' ' * (16 - len(chunk) % 16)).encode("utf-8")
                    outfile.write(encryptor.encrypt(chunk))

    @staticmethod
    def decrypt_enc(in_filename, key, out_filename=None, chunksize=24 * 1024):
        """ Decrypts a file using AES (CBC mode) with the
            given key. Parameters are similar to encrypt_file,
            with one difference: out_filename, if not supplied
            will be in_filename without its last extension
            (i.e. if in_filename is 'aaa.zip.enc' then
            out_filename will be 'aaa.zip')
        """
        if not out_filename:
            out_filename = os.path.splitext(in_filename)[0]

        with open(in_filename, 'rb') as infile:
            origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
            iv = infile.read(16)
            decryptor = AES.new(key, AES.MODE_CBC, iv)

            with open(out_filename, 'wb') as outfile:
                while True:
                    chunk = infile.read(chunksize)
                    if len(chunk) == 0:
                        break
                    outfile.write(decryptor.decrypt(chunk))
                outfile.truncate(origsize)

    def do_process(self):
        """编译"""
        self.do_init()
        self.copy_dir()
        self.get_all_file_need_entrype()

        for index, file_path in enumerate(self.entrypt_file_list):
            print(index, file_path)
            if file_path.endswith('.py'):
                # do encrypt
                if self.entrypt_mode == "so":
                    self.encrypt_so(os.path.splitext(file_path)[0])
                elif self.entrypt_mode == "enc":
                    self.encrypt_enc(file_path)
                else:
                    raise ValueError("entrypt_model can only be so or enc")
                # clear py file
                if self.is_clear:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            elif file_path.endswith('.pyc'):
                os.remove(file_path)


class CompilePyToPyc(object):
    """将制定文件夹中的 py 文件全部编译为 pyc 文件"""

    @staticmethod
    def find_all_py_in_folder(folder_path):
        """找到文件夹中的所有 py 文件"""
        py_files = list()
        for i, j, k in os.walk(folder_path):
            for file_py in k:
                if file_py.endswith('.py'):
                    py_path_temp = os.path.join(i, file_py)
                    py_files.append(py_path_temp)
        return py_files

    @staticmethod
    def compile_and_del(py_files):
        """编译和删除 py 文件"""
        # 对 py 文件进行编译
        for each_py_file in py_files:
            py_compile.compile(each_py_file)

        # 删除编译后的 py 文件
        for each_py_file in py_files:
            os.remove(each_py_file)

    @staticmethod
    def do_compile_for_dir(folder_path):
        """对某一个文件夹进行编译操作？（这样解释似乎不合适）"""
        try:
            # 找到 py 文件
            py_files = CompilePyToPyc.find_all_py_in_folder(folder_path)
            # 编译和删除 py 文件
            CompilePyToPyc.compile_and_del(py_files)
            print('compile for {} file'.format(len(py_files)))
            return True
        except:
            return False



if __name__ == '__main__':

    # fixme 编译为 so 代码目前还不支持在 windows 环境下面使用

    a = EncryptPy()
    a.is_copy = False
    a.is_clear = False

    if len(sys.argv) <= 1:
        a.entrypt_dir_list = [r"C:\Users\14271\Desktop\comtest"]
        a.env_path = r"C:\Users\14271\anaconda3\envs\ldq\pythonw.exe"
        # a.exclude_dir_list = [r"/home/ldq/del/entrypt/Algo/Tree"]
        a.do_process()
    else:
        args = args_parse()

        if args.entrypt_dir:
            a.entrypt_dir_list = args.entrypt_dir.strip().split(',')

        if args.exclude_dir:
            a.exclude_dir_list = args.exclude_dir.strip().split(',')

        if args.entrypt_file:
            a.entrypt_file_list = args.entrypt_file.strip().split(',')

        if args.exclude_file:
            a.exclude_file_list = args.exclude_file.strip().split(',')

        if args.env_path:
            a.env_path = args.env_path

        if args.encrypt_mode:
            a.entrypt_mode = args.encrypt_mode

        if args.clear:
            a.is_clear = True

        if args.old_dir:
            a.is_copy = True

        a.do_process()
