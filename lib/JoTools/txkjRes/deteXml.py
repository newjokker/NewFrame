# -*- coding: utf-8  -*-
# -*- author: jokker -*-


from ..utils.XmlUtil import XmlUtil

# fixme 重写这个函数，速度更快

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
            if node_name in ["name", "pose", "truncated", "difficult", "prob", "id", "des", "crop_path"]:
                object_info[node_name] = XmlUtil.get_info_from_node(each_node)['value']

            # ----------------------------------------------------------------------------------------------------------
            # fixme 符合武汉那边给的规范
            if node_name in ["code", "label"]:
                object_info["name"] = XmlUtil.get_info_from_node(each_node)['value']

            # fixme ** 写的不规范的 xml
            # if node_name in ["angle"]:
            #     object_info["name"] = str(XmlUtil.get_info_from_node(each_node)['value'])
            # ----------------------------------------------------------------------------------------------------------

            elif node_name == "bndbox":
                bndbox_info = {}
                for each_node_2 in each_node.childNodes:
                    each_node_name = each_node_2.nodeName
                    if each_node_name in ["xmin", "ymin", "xmax", "ymax"]:
                        bndbox_info[each_node_name] = XmlUtil.get_info_from_node(each_node_2)['value']

                    # --------------------------------------------------------------------------------------------------
                    # fixme 这边为了兼容 ** 写的不规范的 xml
                    # if each_node_name in ["xMin", "yMin", "xMax", "yMax"]:
                    #     bndbox_info[str(each_node_name).lower()] = XmlUtil.get_info_from_node(each_node_2)['value']
                    # --------------------------------------------------------------------------------------------------

                object_info['bndbox'] = bndbox_info
            elif node_name == "robndbox":
                robndbox_info = {}
                for each_node_2 in each_node.childNodes:
                    each_node_name = each_node_2.nodeName
                    if each_node_name in ["cx", "cy", "w", "h", "angle"]:
                        robndbox_info[each_node_name] = XmlUtil.get_info_from_node(each_node_2)['value']
                object_info['robndbox'] = robndbox_info
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
                if (each_node != "bndbox") and (each_node != "robndbox"):
                    XmlUtil.add_sub_node(root, object_node, each_node, each_object[each_node])
                elif each_node == "bndbox":
                    bndbox_node = XmlUtil.add_sub_node(root, object_node, "bndbox", "")
                    for each_bndbox in each_object["bndbox"]:
                        XmlUtil.add_sub_node(root, bndbox_node, each_bndbox, each_object["bndbox"][each_bndbox])
                else:
                    bndbox_node = XmlUtil.add_sub_node(root, object_node, "robndbox", "")
                    for each_bndbox in each_object["robndbox"]:
                        XmlUtil.add_sub_node(root, bndbox_node, each_bndbox, each_object["robndbox"][each_bndbox])
        # 保存 xml 到文件
        XmlUtil.save_xml(root, save_path)

    def save_to_xml_wh_format(self, save_path, assign_xml_info):
        """将 xml 保存为武汉提供的格式"""
        if assign_xml_info is None:
            assign_xml_info = self.__xml_info_dict.copy()
        # 没有值
        if not assign_xml_info:
            raise ValueError("xml info is empty")
        # 写 xml
        root = XmlUtil.get_document()
        xml_calss_1 = XmlUtil.add_sub_node(root, root, 'annotation', '')
        # 增加 "folder", "filename", "path", "segmented"
        for attr_name in ["filename"]:
            XmlUtil.add_sub_node(root, xml_calss_1, attr_name, assign_xml_info[attr_name])
        # 增加 size
        size_node = XmlUtil.add_sub_node(root, xml_calss_1, "size", '')
        XmlUtil.add_sub_node(root, xml_calss_1, "objectsum", str(len(assign_xml_info["object"])))
        XmlUtil.add_sub_node(root, size_node, "width", str(int(float(assign_xml_info["size"]["width"]))))
        XmlUtil.add_sub_node(root, size_node, "height", assign_xml_info["size"]["height"])
        XmlUtil.add_sub_node(root, size_node, "depth", assign_xml_info["size"]["depth"])
        # 增加 object
        index = 0
        for each_object in assign_xml_info["object"]:
            index += 1
            object_node = XmlUtil.add_sub_node(root, xml_calss_1, "object", '')
            XmlUtil.add_sub_node(root, object_node, "Serial", str(index))
            XmlUtil.add_sub_node(root, object_node, "code", each_object["name"])
            bndbox_node = XmlUtil.add_sub_node(root, object_node, "bndbox", "")
            XmlUtil.add_sub_node(root, bndbox_node, "xmin", each_object["bndbox"]["xmin"])
            XmlUtil.add_sub_node(root, bndbox_node, "ymin", each_object["bndbox"]["ymin"])
            XmlUtil.add_sub_node(root, bndbox_node, "xmax", each_object["bndbox"]["xmax"])
            XmlUtil.add_sub_node(root, bndbox_node, "ymax", each_object["bndbox"]["ymax"])

        # 保存 xml 到文件
        XmlUtil.save_xml(root, save_path)

def parse_xml(xml_path):
    """简易的函数使用版本"""
    a = ParseXml()
    xml_info = a.get_xml_info(xml_path)
    return xml_info

def save_to_xml(xml_info, xml_path):
    """保存为 xml"""
    a = ParseXml()
    a.save_to_xml(save_path=xml_path, assign_xml_info=xml_info)

def save_to_xml_wh_format(xml_info, xml_path):
    """按照武汉的格式保存 xml """
    a = ParseXml()
    a.save_to_xml_wh_format(save_path=xml_path, assign_xml_info=xml_info)

def parse_xml_as_txt(xml_path):
    """使用读取存文本的方式读取 xml """

    def parse_assign_line(each_xml_line, assign_tag):
        """解析指定行中的指定标签"""
        return each_xml_line.strip()[len(assign_tag) + 2: -len(assign_tag) - 3]

    xml_info = {'size': {'height': -1, 'width': -1, 'depth': -1},
                'filename': '', 'path': '', 'object': [], 'folder': '',
                'segmented': '', 'source': ''}

    with open(xml_path, 'r', encoding='utf-8') as xml_file:
        each_line = next(xml_file)
        while each_line:
            each_line = each_line.strip()

            if each_line.startswith('<filename>'):
                xml_info['filename'] = parse_assign_line(each_line, 'filename')
            elif each_line.startswith('<folder>'):
                xml_info['folder'] = parse_assign_line(each_line, 'folder')
            elif each_line.startswith('<height>'):
                xml_info['height'] = float(parse_assign_line(each_line, 'height'))
            elif each_line.startswith('<width>'):
                xml_info['width'] = float(parse_assign_line(each_line, 'width'))
            elif each_line.startswith('<depth>'):
                xml_info['depth'] = float(parse_assign_line(each_line, 'depth'))
            elif each_line.startswith('<path>'):
                xml_info['path'] = parse_assign_line(each_line, 'path')
            elif each_line.startswith('<segmented>'):
                xml_info['segmented'] = parse_assign_line(each_line, 'segmented')
            elif each_line.startswith('<source>'):
                xml_info['source'] = parse_assign_line(each_line, 'source')
            elif each_line.startswith('<object>'):
                each_obj = {'name': '', 'prob': -1, 'id':-1, 'des':'','crop_path':'',
                            'bndbox': {'xmin': -1, 'xmax': -1, 'ymin': -1, 'ymax': -1}}
                while True:
                    each_line = next(xml_file)
                    each_line = each_line.strip()

                    if each_line.startswith('</object>'):
                        xml_info['object'].append(each_obj)
                        break
                    elif each_line.startswith('<name>'):
                        each_obj['name'] = parse_assign_line(each_line, 'name')
                    elif each_line.startswith('<prob>'):
                        each_obj['prob'] = float(parse_assign_line(each_line, 'prob'))
                    elif each_line.startswith('<id>'):
                        each_obj['id'] = float(parse_assign_line(each_line, 'id'))
                    elif each_line.startswith('<des>'):
                        each_obj['des'] = parse_assign_line(each_line, 'des')
                    elif each_line.startswith('<crop_path>'):
                        each_obj['crop_path'] = parse_assign_line(each_line, 'crop_path')
                    elif each_line.startswith('<xmin>'):
                        each_obj['bndbox']['xmin'] = float(parse_assign_line(each_line, 'xmin'))
                    elif each_line.startswith('<xmax>'):
                        each_obj['bndbox']['xmax'] = float(parse_assign_line(each_line, 'xmax'))
                    elif each_line.startswith('<ymin>'):
                        each_obj['bndbox']['ymin'] = float(parse_assign_line(each_line, 'ymin'))
                    elif each_line.startswith('<ymax>'):
                        each_obj['bndbox']['ymax'] = float(parse_assign_line(each_line, 'ymax'))

            elif each_line.startswith('</annotation>'):
                return xml_info

            each_line = next(xml_file)

