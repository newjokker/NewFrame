# -*- coding: utf-8  -*-
# -*- author: jokker -*-

from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.CsvUtil import CsvUtil
from JoTools.utils.FileOperationUtil import FileOperationUtil
import os

# 映射字典
tag_code_dict = {

    # --------------------------------------------------------------------------------------------------------------
    # 开口销缺失
    "K": "040500013",

    # 安装不规范
    "illegal": "040500023",

    # 销钉锈蚀
    "K_KG_rust": "040500033",

    # 螺母锈蚀
    "Lm_rust": "040501013",
    # --------------------------------------------------------------------------------------------------------------

    # 鸟巢蜂巢
    "nc": "010000023",
    "nest": "010000023",

    # 玻璃绝缘子自爆
    "jyzzb": "030100023",

    # 绝缘子污秽
    "abnormal": "030100011",

    # 均压环倾斜
    "fail": "030200131",

    # 金具锈蚀
    "rust": "040000011",

    # 防振锤锈蚀
    "fzc_rust": "040303031",

    # 防振锤破损
    "fzc_broken": "040303021",

    # fixme 导线散股,看看这个标签是否正确
    "sg": "040402011",

    # --------------------------------------------------------------------------------------------------------------
    # 吊塔
    "TowerCrane": "060800013",

    # 推土机
    "Bulldozer": "060800023",

    # 挖掘机
    "Digger": "060800033",

    "CementPumpTruck_yb": "060800033",
    # --------------------------------------------------------------------------------------------------------------

    # 线夹缺垫片
    "dp_missed": "040001042",

    # 防鸟刺安装不规范
    "fncBGF": "070400031",

    # 防鸟刺未打开
    "weidakai": "070400021",

                 }


def xml_to_csv(xml_dir, csv_save_path):
    """将保存的 xml 文件信息存放在 csv 文件中"""
    csv_list = [['filename', 'code', 'score', 'xmin', 'ymin', 'xmax', 'ymax']]
    #
    for each_xml_path in FileOperationUtil.re_all_file(xml_dir, endswitch=['.xml']):
        try:
            dete_res = DeteRes(xml_path=each_xml_path)
            # 输出对应的 csv，filename,code,score,xmin,ymin,xmax,ymax
            for dete_obj in dete_res:
                csv_list.append([each_xml_path[:-3] + 'jpg', dete_obj.tag, dete_obj.conf, dete_obj.x1, dete_obj.y1, dete_obj.x2, dete_obj.y2])
        except Exception as e:
            print('-' * 100)
            print('GOT ERROR---->')
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)
    CsvUtil.save_list_to_csv(csv_list, csv_save_path)


def merge_xml_list(xml_path_list, save_path):
    """将 xml 进行合并，获取 DeteRes"""
    if len(xml_path_list) == 1:
        a = DeteRes(xml_path=xml_path_list[0])
    elif len(xml_path_list) > 1:
        a = DeteRes(xml_path=xml_path_list[0])
        for each_assign_xml_path in xml_path_list[1:]:
            each_dete_res = DeteRes(xml_path=each_assign_xml_path)
            a += each_dete_res
    else:
        return None
    # 使用映射字典进行映射
    a.update_tags(tag_code_dict)
    # fixme 这边要保存为武汉的格式，直接写一个函数
    a.save_to_xml(save_path, format='wuhan')


def merge_xml(xml_dir_list, save_dir):
    """将 xml_dir 中文件名相同的 xml 进行合并，放到 save_dir 文件夹下"""

    # 拿到 xml 文件夹路径列表中的所有 xml 路径
    xml_path_list = []
    for each_xml_dir in xml_dir_list:
        for each_xml_path in FileOperationUtil.re_all_file(each_xml_dir, endswitch=['.xml']):
            xml_path_list.append(each_xml_path)

    # 合并
    xml_dict = {}
    for each_xml_path in xml_path_list:
        each_xml_name = os.path.split(each_xml_path)[1]
        if each_xml_name in xml_dict:
            xml_dict[each_xml_name].append(each_xml_path)
        else:
            xml_dict[each_xml_name] = [each_xml_path]
    # 合并
    for each_xml_name, each_xml_path_list in xml_dict.items():
        save_xml_path = os.path.join(save_dir, each_xml_name)
        merge_xml_list(each_xml_path_list, save_xml_path)


if __name__ == "__main__":

    region_xml_dir_list = [r"C:\Users\14271\Desktop\del\Annotations"]
    merge_xml_dir = r"C:\Users\14271\Desktop\del\Annotations_wuhan"
    csv_path = r"C:\Users\14271\Desktop\del\Annotations_wuhan.csv"

    # 合并 xml
    merge_xml(region_xml_dir_list, merge_xml_dir)
    # 保存为 csv 文件
    xml_to_csv(merge_xml_dir, csv_path)






