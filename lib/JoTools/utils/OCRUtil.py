# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 参考 ： https://blog.csdn.net/jclian91/article/details/80628188

# 解决报错的问题 ：https://www.cnblogs.com/benson321/p/10502339.html

# 用法：

"""
图片清晰一点，识别率还是挺高的，可以将之前的段子找到，全部数字化
"""

import pytesseract
from PIL import Image
from pytesseract import image_to_boxes, image_to_data
from pytesseract import image_to_osd, Output

# todo 看看能不能拿到每个字的具体位置

class OCRUtil(object):

    @staticmethod
    def get_words_from_image(img_path, lang='chi_sim', tesseract_cmd=None):
        """从图片中识别文字"""
        try:
            if tesseract_cmd is None:
                tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            text = pytesseract.image_to_string(Image.open(img_path), lang=lang, output_type=Output.DICT)  # chinese
            # # text = pytesseract.image_to_string(Image.open(img_path))  # chinese
            # a = image_to_boxes(Image.open(img_path))
            # b = image_to_osd(Image.open(img_path))
            # c = image_to_data(Image.open(img_path), output_type=Output.DICT)
            return text
        except TypeError:
            return None


if __name__ == '__main__':

    # todo 写一个带界面的程序，在照片上画可以倾斜的矩形，识别其中的文字

    Text = OCRUtil.get_words_from_image(r'C:\Users\14271\Desktop\test.png')

    print(Text)
