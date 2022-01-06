# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import cv2
import numpy as np
# from JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteRes import DeteRes
import matplotlib.pyplot as plt


img_path = r"C:\Users\14271\Desktop\del\test\00fa186e8d4d6660b49ddef8a35a77de.jpg"
xml_path = r"C:\Users\14271\Desktop\del\test\save.xml"
save_xml_path = r"C:\Users\14271\Desktop\del\test\save_002.xml"

a = DeteRes(xml_path=xml_path)
a.img_path = img_path

print(a.des)









