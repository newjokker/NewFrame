# -*- coding: utf-8  -*-

import os
from Crypto.Cipher import AES
import random, struct
import configparser
import argparse
import sys

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='./')
    parser.add_argument('--model_name', dest='model_name', type=str)
    parser.add_argument('--suffix', dest='suffix', type=str, default='.pth')
    args = parser.parse_args()
    return args

def encrypt_file(key, in_filename, out_filename=None, chunksize=64*1024):
    """ Encrypts a file using AES (CBC mode) with the
        given key.

        key:
            The encryption key - a string that must be
            either 16, 24 or 32 bytes long. Longer keys
            are more secure.

        in_filename:
            Name of the input file

        out_filename:
            If None, '<in_filename>.enc' will be used.

        chunksize:
            Sets the size of the chunk which the function
            uses to read and encrypt the file. Larger chunk
            sizes can be faster for some files and machines.
            chunksize must be divisible by 16.
    """
    if not out_filename:
        out_filename = in_filename + '.enc'

    #iv = ''.join(chr(random.randint(0, 0xFF)) for i in range(16))
    #encryptor = AES.new(key, AES.MODE_CBC, iv)
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

def decrypt_file(key, in_filename, out_filename=None, chunksize=24 * 1024):
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

def readConfig(configPath):
    model_name_list = []

    cf = configparser.ConfigParser()
    cf.read(configPath)
    sections = cf.sections()
    for s in sections:
        try:
            model_name_list.append(cf.get(s, 'modelname'))

        except:
            continue

    return model_name_list

def modifyEncryptStatus(configPath,isEncrypt):
    model_name_list = []

    cf = configparser.ConfigParser()
    cf.read(configPath)
    cf.set('common', 'encryption', str(isEncrypt))
    with open(configPath, 'w') as f:
        cf.write(f)
    return 

def single_model_encrypt(models_dir, name, ext):
    src_file = os.path.join(models_dir, name+ext)
    if not os.path.exists(src_file):
       return
    else:
        dst_file = os.path.join(models_dir, name + '_locked' + ext)  # 加密后的模型
        encrypt_file(bkey32, src_file, dst_file)  # 加密   ### key, in_filename, out_filename

def single_model_decrypt(models_dir, name, ext):
    src_file = os.path.join(models_dir, name+ '_locked' +ext)
    if not os.path.exists(src_file):
       return
    else:
        dst_file = os.path.join(models_dir, name +  ext)  # 加密后的模型
        decrypt_file(bkey32, src_file, dst_file)  # 加密   ### key, in_filename, out_filename

def single_project_models_decrypt(project_dir):
    models_dir = os.path.join(project_dir, 'models')
    config_path = os.path.join(project_dir, 'config.ini')
    models = os.listdir(models_dir)
    dstmodels = [m for m in models if '_locked' in m]
    for m in dstmodels:
        name, ext = m.split('_locked')
        print("=============== 解密模型 {} ===============".format(m))
        single_model_decrypt(models_dir, name, ext)
    modifyEncryptStatus(config_path,False)    

def single_project_models_encrypt(project_dir):
    config_path = os.path.join(project_dir, 'config.ini')
    models_dir  = os.path.join(project_dir, 'models')

    if os.path.exists(config_path) and os.path.exists(models_dir):
        model_list = readConfig(config_path)
        for model_name in model_list:
            name, ext = os.path.splitext(model_name)
            if   ext == '.model' or ext == '.pb':
                print("=============== 加密 {} ===============".format(model_name))
                single_model_encrypt(models_dir, name, ext)

            elif ext == '.ckpt':
                print("=============== 加密 {} ===============".format(model_name))
                suffixs = [".meta", ".index", ".data-00000-of-00001"]
                for sf in suffixs:
                    new_sf = ext+sf
                    single_model_encrypt(models_dir, name, new_sf)

            else:
                print("!!!!!!!!! wrong ext model: {} !!!!!!!!!".format(model_name))
        modifyEncryptStatus(config_path,True)
    else:
        print('model dir or config.ini do not exist')

def single_project_srcmodels_clear(project_dir):
    config_path = os.path.join(project_dir, 'config.ini')
    models_dir  = os.path.join(project_dir, 'models')

    if os.path.exists(config_path) and os.path.exists(models_dir):
        model_list = readConfig(config_path)
        for model_name in model_list:
            name, ext = os.path.splitext(model_name)
            if ext == '.model' or ext == '.pb':
                print("=============== 删除模型 {} ===============".format(model_name))
                model_path = os.path.join(models_dir, model_name)
                os.remove(model_path)

            elif ext == '.ckpt':
                print("=============== 删除模型 {} ===============".format(model_name))
                suffixs = [".meta", ".index", ".data-00000-of-00001"]
                for sf in suffixs:
                    new_sf = ext + sf
                    model_path = os.path.join(models_dir, name+new_sf)
                    os.remove(model_path)

            else:
                print("!!!!!!!!! wrong ext model: {} !!!!!!!!!".format(model_name))

    else:
        print('!!!!!!!!! model dir or config.ini do not exist !!!!!!!!!')

def single_project_dstmodels_clear(project_dir):
    models_dir  = os.path.join(project_dir, 'models')

    models = os.listdir(models_dir)
    dstmodels = [m for m in models if '_locked' in m]
    for m in dstmodels:
        print("=============== 删除模型 {} ===============".format(m))
        model_path = os.path.join(models_dir, m)
        os.remove(model_path)

def main(bgl_dir, project_list, use_operation):
    """对所有文件进行处理"""

    if not os.path.exists(bgl_dir):
        print('!!!!!!!!! wrong BGL path !!!!!!!!!')
        return

    for project in project_list:
        # 工程文件夹
        project_dir = os.path.join(bgl_dir, project)
        if not os.path.exists(project_dir):
            print('!!!!!!!!! project {} dir does not exists !!!!!!!!!'.format(project))
            continue

        else:
            # 版本文件夹
            l = os.listdir(project_dir)
            l = [i for i in l if os.path.isdir(os.path.join(project_dir, i)) and not i.endswith('-bak')]
            if l:
                for i in l:
                    version = i
                    print('======== {} models from project {}({}) ========'.format(op_name[args.operation], project, version))
                    version_project_dir = os.path.join(project_dir, version)
                    # 使用传入的方法对版本文件夹进行操作
                    use_operation(version_project_dir)
            else:
                print('!!!!!!!!! no version folder !!!!!!!!!')


if __name__ == "__main__":


    salt = 'txkj2019'
    bkey32 = "{: <32}".format(salt).encode("utf-8")

    args = args_parse()

    model_dir = args.model_dir
    model_name = args.model_name
    suffix = args.suffix
    single_model_encrypt(model_dir, model_name, suffix)



