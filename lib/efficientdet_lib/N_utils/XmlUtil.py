# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import xml.dom.minidom
from xml.dom.minidom import Document


class XmlUtil(object):

    # ------------------------------ write xml -------------------------------------------------------------------------
    @staticmethod
    def get_document(head_time_str='xml'):
        """return document for xml writing"""
        document = Document()
        document.appendChild(document.createComment(head_time_str))
        return document

    @staticmethod
    def add_sub_node(document, cur_node, node_key, node_value, node_att=None):
        """add sub node"""
        if node_att is None:
            node_att = {}
        try:
            child = document.createElement(node_key)
            # write attribute
            for attKey in node_att:
                child.setAttribute(attKey, node_att[attKey])
            # write value
            if node_value:
                child_text = document.createTextNode(node_value)
                child.appendChild(child_text)
            cur_node.appendChild(child)
            # add node
            return child
        except:
            print("* error in add node")
            return None

    @staticmethod
    def save_xml(document, xml_path):
        """save to xml file"""
        with open(xml_path, 'wb') as f:  # python 2|3 ==> 'w' | 'wb'
            f.write(document.toprettyxml(indent='\t', encoding='utf-8'))

    # ------------------------------ read xml --------------------------------------------------------------------------
    @staticmethod
    def get_root_node(xml_path):
        """return collection of xml"""
        dom_tree = xml.dom.minidom.parse(xml_path)
        root_node = dom_tree.documentElement
        return root_node

    @staticmethod
    def get_info_from_node(each_node, assign_attr=None):
        """ get_info_from_node, only support Element now"""
        # -----------------------------------------------------------------
        if each_node.nodeType != 1:
            return
        # -----------------------------------------------------------------
        element_info = {}
        # -----------------------------------------------------------------
        # get all attribute
        attr_dict = {}
        if assign_attr:
            assign_attr = set(assign_attr)
            for each_attr in assign_attr:
                attr_dict[each_attr] = each_node.getAttribute(each_attr)
        else:
            for each_attr in each_node.attributes.values():
                attr_dict[each_attr.name] = each_attr.value
        element_info['attr'] = attr_dict
        # -----------------------------------------------------------------
        # if node have child node_vale is None
        node_value = None
        if len(each_node.childNodes) == 1:
            if each_node.childNodes[0].nodeType == 3:
                node_value = each_node.childNodes[0].nodeValue
        element_info['value'] = node_value
        # -----------------------------------------------------------------
        # get child node
        # child_nodes = eachNode.childNodes
        # -----------------------------------------------------------------
        return element_info

    # ------------------------------ 常用 ------------------------------------------------------------------------------
    @staticmethod
    def xml_parser(xml_path, need_tages, attr="identify"):
        """
        parse xml file to dict
        :param xml_path: xml path ==> str
        :param need_tages: need tags ==> list
        :param attr: needed attribute name(be key of result dict) ==> str
        :return: {'attr':value}
        """

        def get_key_value(one_node, attr_temp):
            """read tag"""
            key = one_node.getAttribute(attr_temp)
            value = one_node.childNodes[0].data
            return key, value

        xml_info = {}
        dom_tree = xml.dom.minidom.parse(xml_path)
        collection = dom_tree.documentElement
        # loop node
        for each_tag in need_tages:
            for eachNode in collection.getElementsByTagName(each_tag):
                (info_key, info_value) = get_key_value(eachNode, attr)
                xml_info[info_key] = info_value
        return xml_info


if __name__ == '__main__':

    xmlPath = r'D:\Code\Util_Util\Z_other\DuanZhi\AuxData\jokker.xml'

    root = XmlUtil.get_root_node(xmlPath)

    for each in root.childNodes:

        if XmlUtil.get_info_from_node(each):
            # print  Xml_Util.get_info_from_node(each)['value']
            print(XmlUtil.get_info_from_node(each))

"""
1. nodeType 的意思，不同的数字代表不同的值，其中前三个为，1：Element， 2：Attribute， 3：Text
"""
