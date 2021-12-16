# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 参考：https://blog.csdn.net/qq_35952638/article/details/95088227

"""
* 经测试，原数据可以很大，所以完全可以使用原数据对图片打标签
"""

from pyexiv2 import Image


class ImageMetadataUtil(object):
    """图像原数据操作"""

    @staticmethod
    def add_key_to_img(img_path, key_words):
        """往图像中写入标签数据，可以以逗号分隔，或者用 json 格式"""

        if isinstance(key_words, list) or isinstance(key_words, tuple) or isinstance(key_words, set):
            key_words = map(lambda x:str(x), key_words)
            key_words = ",".join(key_words)

        key_words_dict = {'Iptc.Application2.Keywords': key_words}
        img = Image(img_path)
        img.modify_iptc(key_words_dict)
        img.close()

    @staticmethod
    def read_key_from_img(img_path):
        """拿到图像标签数据"""
        img = Image(img_path)
        img_info_dict = img.read_iptc()
        img.close()
        if "Iptc.Application2.Keywords" in img_info_dict:
            key_words = img_info_dict["Iptc.Application2.Keywords"]
            return key_words
        else:
            return None


if __name__ == "__main__":

    img_path = r"C:\Users\14271\Desktop\test.jpg"

    ImageMetadataUtil.add_key_to_img(img_path, [1,2,3,4,5])

    a = ImageMetadataUtil.read_key_from_img(img_path)

    print(a)


