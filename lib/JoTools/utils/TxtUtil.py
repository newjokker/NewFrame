# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os


class TxtUtil(object):
    """Txt操作"""

    @staticmethod
    def read_as_tableread_as_table(txt_path, sep='\t'):
        return_table = []
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            for each_line in txt_file:
                line_elements = []
                for each_element in each_line.split(sep=sep):
                    line_elements.append(each_element.strip('\n'))
                return_table.append(line_elements)
        return return_table

    @staticmethod
    def write_table_to_txt(table, txt_path, sep='\t', end_line=''):
        with open(txt_path, 'w') as txt_file:
            for each_line in table:
                each_line = map(lambda x:str(x), each_line)
                txt_file.write(sep.join(each_line) + end_line)

    # ------------------------------------ need repair -----------------------------------------------------------------

    @staticmethod
    def merge_txt(folder_path, save_txt):
        """txt合成"""
        lines = []
        for i, j, k in os.walk(folder_path):
            for each_file_name in k:
                # find txt
                if not each_file_name.endswith('.txt'):
                    continue

                # 读取数据
                abs_path = os.path.join(folder_path, each_file_name)
                lines.append('\n' + '-' * 100 + '\n')
                lines.append('{0}\n'.format(abs_path))

                with open(abs_path, 'r') as txt_file:
                    lines.extend(txt_file.readlines())

        # write txt
        with open(save_txt, 'w') as txt_file:
            for each in lines:
                txt_file.write(each)


if __name__ == "__main__":

    with open(r"C:\data\load\train_image_id_from_imagenet.txt", "r") as txt_file:
        for each in txt_file:
            print(each)



    pass
