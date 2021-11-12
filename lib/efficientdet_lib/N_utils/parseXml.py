# -*- coding: utf-8  -*-
# -*- author: jokker -*-

from .XmlUtil import XmlUtil


class ParseXml(object):
    """解析 xml 中的信息，将信息导出为 xml"""

    def __init__(self):
        self.__attrs = {"folder", "filename", "path", "segmented", "size", "source", "object"}  # 所有的属性
        self.__xml_info_dict = {}  # xml 信息字典
        self.__objects_info = []
        self.__size_info = {}
        self.__source_info = {}

    def _parse_node(self, assign_node):
        """解析在字典中的关键字"""
        node_name = assign_node.nodeName
        element_info = XmlUtil.get_info_from_node(assign_node)
        self.__xml_info_dict[node_name] = element_info['value']

    def _parse_object(self, assign_node):
        """解析 object 中的数据"""
        object_info = {}
        for each_node in assign_node.childNodes:
            node_name = each_node.nodeName
            if node_name in ["name", "pose", "truncated", "difficult"]:
                object_info[node_name] = XmlUtil.get_info_from_node(each_node)['value']
            elif node_name == "bndbox":
                bndbox_info = {}
                for each_node_2 in each_node.childNodes:
                    each_node_name = each_node_2.nodeName
                    if each_node_name in ["xmin", "ymin", "xmax", "ymax"]:
                        bndbox_info[each_node_name] = XmlUtil.get_info_from_node(each_node_2)['value']
                object_info['bndbox'] = bndbox_info
        self.__objects_info.append(object_info)

    def _parse_size(self, assign_node):
        """解析 size 信息"""
        for each_node in assign_node.childNodes:
            node_name = each_node.nodeName
            if node_name in ["width", "height", "depth"]:
                self.__size_info[node_name] = XmlUtil.get_info_from_node(each_node)['value']

    def _parse_source(self, assign_node):
        """解析 source 信息"""
        for each_node in assign_node.childNodes:
            node_name = each_node.nodeName
            if node_name in ["database"]:
                self.__source_info[node_name] = XmlUtil.get_info_from_node(each_node)['value']

    def _parse_xml(self, xml_path):
        """解析 xml"""
        root_node = XmlUtil.get_root_node(xml_path)  # 得到根节点
        # 遍历根节点下面的子节点
        for each_node in root_node.childNodes:
            node_name = each_node.nodeName
            if node_name in ["folder", "filename", "path", "segmented"]:
                self._parse_node(each_node)
            elif node_name == "source":
                self._parse_source(each_node)
            elif node_name == "size":
                self._parse_size(each_node)
            elif node_name == "object":
                self._parse_object(each_node)

    def set_attr_info(self, attr, info):
        """设置属性值"""
        if attr not in self.__attrs:
            raise ValueError("""attr should in folder, filename, path, segmented, size, source, object""")
        self.__xml_info_dict[attr] = info

    def update_xml_info(self, up_info):
        """更新 xml 字典信息，up_info: dict"""
        for each_attr in up_info:
            if each_attr not in self.__attrs:
                raise ValueError("""attr should in folder, filename, path, segmented, size, source, object""")
            else:
                self.__xml_info_dict[each_attr] = up_info[each_attr]

    def get_xml_info(self, xml_path):
        # 解析 xml
        self.__xml_info_dict = {"folder": None, "filename": None, "path": None, "segmented": None}
        self._parse_xml(xml_path)
        # 将 xml 中的信息整理输出
        self.__xml_info_dict['size'] = self.__size_info
        self.__xml_info_dict['source'] = self.__source_info
        self.__xml_info_dict['object'] = self.__objects_info
        return self.__xml_info_dict

    def save_to_xml(self, save_path, assign_xml_info=None):
        """将 xml_info 保存为 xml 形式"""
        if assign_xml_info is None:
            assign_xml_info = self.__xml_info_dict.copy()
        # 没有值
        if not assign_xml_info:
            raise ValueError("xml info is empty")
        # 写 xml
        root = XmlUtil.get_document()
        xml_calss_1 = XmlUtil.add_sub_node(root, root, 'annotation', '')
        # 增加 "folder", "filename", "path", "segmented"
        for attr_name in ["folder", "filename", "path", "segmented"]:
            XmlUtil.add_sub_node(root, xml_calss_1, attr_name, assign_xml_info[attr_name])
        # 增加 source
        source_node = XmlUtil.add_sub_node(root, xml_calss_1, "source", '')
        for each_node in assign_xml_info["source"]:
            XmlUtil.add_sub_node(root, source_node, each_node, assign_xml_info["source"][each_node])
        # 增加 size
        size_node = XmlUtil.add_sub_node(root, xml_calss_1, "size", '')
        for each_node in assign_xml_info["size"]:
            XmlUtil.add_sub_node(root, size_node, each_node, assign_xml_info["size"][each_node])
        # 增加 object
        for each_object in assign_xml_info["object"]:
            object_node = XmlUtil.add_sub_node(root, xml_calss_1, "object", '')
            for each_node in each_object:
                if each_node != "bndbox":
                    XmlUtil.add_sub_node(root, object_node, each_node, each_object[each_node])
                else:
                    bndbox_node = XmlUtil.add_sub_node(root, object_node, "bndbox", "")
                    for each_bndbox in each_object["bndbox"]:
                        XmlUtil.add_sub_node(root, bndbox_node, each_bndbox, each_object["bndbox"][each_bndbox])
        # 保存 xml 到文件
        XmlUtil.save_xml(root, save_path)


def parse_xml(xml_path):
    """简易的函数使用版本"""
    a = ParseXml()
    xml_info = a.get_xml_info(xml_path)
    return xml_info


if __name__ == "__main__":

    xml_path = r"C:\Users\14271\Desktop\merge_w_divice_h_1.8\110kV远布一线二线_#138塔_其它（导地线、光缆、基础、接地装置、杆号牌及附属设施等）_地线_DJI_0585.xml"
    xmlInfo = parse_xml(xml_path)

    for each in xmlInfo['object']:
        print(each['name'])




