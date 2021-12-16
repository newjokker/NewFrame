# -*- coding: utf-8  -*-
# -*- author: jokker -*-


from cryptography.fernet import Fernet


class EncryptUtil(object):

    @staticmethod
    def encrypt_file(file_path, save_path):
        key = Fernet.generate_key()
        f = Fernet(key)
        #
        with open(file_path, "rb") as normal_file:
            token = f.encrypt(normal_file.read())
        #
        with open(save_path, "wb") as save_file:
            save_file.write(token)
        return key

    @staticmethod
    def decrypt_file(encrypt_file_path, save_path, key):
        f = Fernet(key)
        #
        with open(encrypt_file_path, "rb") as encrypt_file:
            normal = f.decrypt(encrypt_file.read())
        #
        with open(save_path, "wb") as normal_file:
            normal_file.write(normal)


if __name__ == "__main__":


    img_path = r"C:\Users\14271\Desktop\rustDebug\test.xml"
    save_path = r"C:\Users\14271\Desktop\rustDebug\test_encrypt.xml"
    save_path2 = r"C:\Users\14271\Desktop\rustDebug\test_encrypt2.xml"


    key = EncryptUtil.encrypt_file(img_path, save_path)

    EncryptUtil.decrypt_file(save_path, save_path2, key)









