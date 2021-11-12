import os
import cv2
import xml.etree.ElementTree as ET
import time
import numpy as np
import random
from shapely.geometry import Polygon


def resultConstruction(name, w, h, start, alarms):
    result = {}
    result['filename'] = name + '.jpg'
    result['start_time'] = start
    result['end_time'] = time.time()
    result['width'] = w
    result['height'] = h
    result['alarms'] = alarms
    result['count'] = len(alarms)
    return result


def ratios(step_script, detr=0.15, clsr=0.05):
    step_type = step_script.split("_")[-1]
    if  step_type == "predict":
        ratio = clsr
    else:
        ratio = detr
    return ratio


def parseTrack_simple(track):
    import collections

    flow = collections.OrderedDict()
    for step_script in track.keys():
        models = track[step_script].split(',')
        plans = []

        for model in models:
            model, input_str = model.split('(')
            input = input_str[:-1].split(',')
            plan = {model: input}
            plans.append(plan)

        flow[step_script] = plans
    return flow

def parseTrack(track):
    import collections

    flow = collections.OrderedDict()
    for step_script in track.keys():
        models = track[step_script].split(',')
        scripts = step_script.split(',')
        if len(models)!=len(scripts):
            return None
        plans = []
        for i in range(len(models)):
            model,input_str = models[i].split('(')
            inputData = input_str[:-1].split(':')
            script = scripts[i]
            plan = {'script':script, 'model':model, 'input':inputData}
            plans.append(plan)
        flow[step_script] = plans
    return flow


def check_record_file_path(record_path):
    os.makedirs(record_path, exist_ok=True)


def delTmpFiles(data_path, name):
    for i in os.listdir(data_path):
        file_path = os.path.join(data_path, i)
        if os.path.isfile(file_path) and name in i:
            os.remove(file_path)

        elif os.path.isdir(file_path):
            delTmpFiles(file_path, name)


def free_port(main_port, num):
    import random
    import socket

    try:
        random.seed()
        ports = []
        port_list = list(range(20000, 20100))
        for port in port_list:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = s.connect_ex(('127.0.0.1', port))
            if result and port != main_port:
                ports.append(port)

        return random.sample(ports, num)
    except Exception as e:
        raise e

def getPortForObj_simple(main_port, allflow):
    flows = []
    for _, v in allflow.items():
        flows.extend(v)

    ports = free_port(main_port, len(flows))
    info = {}
    for i, flow in enumerate(flows):
        obj_name = list(flow.keys())[0]
        info[obj_name] = ports[i]
    return info


def getPortForObj(main_port, allflow):
    flows = []
    for _, v in allflow.items():
        flows.extend(v)

    ports = free_port(main_port, len(flows))
    info = {}
    for i, flow in enumerate(flows):
        obj_name = flow['model']
        info[obj_name] = ports[i]
    return info

def getResizedImgSize(im,xmin,xmax,ymin,ymax,xExtend,yExtend):
    width = im.shape[1]
    height = im.shape[0]
    xmin_r = max(xmin - xExtend,0)
    xmax_r = min(xmax + xExtend,width)
    ymin_r = max(ymin - yExtend,0)
    ymax_r = min(ymax + yExtend,height)
    return ymin_r,ymax_r,xmin_r,xmax_r

def getResizedImgSizePro(im,xmin,xmax,ymin,ymax,R):
    width = im.shape[1]
    height = im.shape[0]
    nx = 0
    ny = 0
    D = R * 2
    if(ymax-ymin > D or xmax-xmin > D):
        if(ymax-ymin - (xmax-xmin)> 0):
            R = (ymax-ymin)/2
            D = ymax-ymin
            ny = ymin + R
        elif(xmax-xmin - (ymax-ymin)>0):
            R = (xmax-xmin)/2
            D = xmax -xmin
            nx = xmin + R
    nx = cx = (xmin + xmax) / 2
    ny = cy = (ymin + ymax) / 2
    if(cx + R < width and cx - R > 0 and cy + R < height and cy - R > 0):
        nymin,nymax,nxmin,nxmax = int(cy-R),int(cy+R),int(cx-R),int(cx+R)
        return (nymin,nymax,nxmin,nxmax)
    if(cx +R >width):
        nx = width - R
    if(cx -R<0):
        nx =  R
 
def parse_xml(filename):
    boxes=[]
    tree = ET.parse(filename)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([ymin,xmin,ymax,xmax])
    return boxes
    # save complete resizedXml 


def mat_inter(box1,box2):
    x01,y01,x02,y02 = box1
    x11,y11,x12,y12 = box2
    lx = abs((x01+x02)/2-(x11+x12)/2)
    ly = abs((y01+y02)/2-(y11+y12)/2)
    sax = abs(x01-x02)
    sbx = abs(x11-x12)
    say = abs(y01-y02)
    sby = abs(y11-y12)
    if lx <= (sax+sbx)/2 and ly <= (say + sby)/2:
        return True
    else:
        return False


def compute_iou(box1,box2):
    if mat_inter(box1,box2) != True:
        return 0.0
    x01,y01,x02,y02 = box1
    x11,y11,x12,y12 = box2
    col = min(x02,x12) - max(x01,x11)
    row = min(y02,y12) - max(y01,y11)
    intersection = col * row
    area1 = (x02-x01)*(y02-y01)
    area2 = (x12-x11)*(y12-y11)
    coincide = intersection/(area1+area2-intersection)
    return coincide


def compute_iou_min(box1, box2):
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / min(area1, area2)
        return coincide


def solve_coincide(box1,box2,iouThresh=0.6):
    if compute_iou(box1,box2) > iouThresh:
        return True
    else:
        x01,y01,x02,y02 = box1
        x11,y11,x12,y12 = box2
        mx0 = (x01 + x02) / 2
        my0 = (y01 + y02) / 2
        mx1 = (x11 + x12) / 2
        my1 = (y11 + y12) / 2
        if(isInside(x01,y01,x02,y02,mx1,my1) or isInside(x11,y11,x12,y12,mx0,my0)):
            return True
        else:
            return False


def isInside(x1, y1, x2, y2, x, y):
        if x <= x1 or x >= x2 or y >= y2 or y <= y1:
            return False
        return True


def isBoxExist(boxList,bbox,iouThresh=0.6):
    for box in boxList:
        if solve_coincide(box,bbox,iouThresh):
            return True
        else:
            continue
    return False


def prettyXml(element, indent, newline, level = 0):
    if len(element) != 0:
        if element.text == None or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level = level + 1)


def list2dict(objects):
    obj_dict = {}
    for obj in objects:
        label, i, xmin, ymin, xmax, ymax, p = obj
        obj_dict[str((label, i))] = [xmin, ymin, xmax, ymax, p]
    return obj_dict


def getListFromStr(boxes):
    result = []
    for box in boxes:
        result.append(eval(box))
    return result


def isBoxIn(boxList,bbox):
    for box in boxList:
        if(solve_coincide(box,bbox,0.8)):
            return True
    return False

def sort_obj_by_name(kkxBox):
    kkxBox_sorted = []
    kkxName_list = []
    for kkxObj in kkxBox:
        kkxName = kkxObj['kkxName']
        kkxName_list.append(kkxName)
    kkxName_list.sort()
    for kkxObj in kkxBox:
        kkxName = kkxObj['kkxName']
        for kkx_name in kkxName_list:
            if kkxName == kkx_name:
                kkxBox_sorted.append(kkxObj)

    return  kkxBox_sorted

def save_crop_img(img, img_name, obj_range, obj_label, save_dir,ii):
    """保存截图，返回存储路径"""
    save_name = "{0}_{1}_{2}.jpg".format(img_name[:-4], obj_label,ii)
    save_path = os.path.join(save_dir, save_name)
    each_crop = img.crop(obj_range)
    each_crop.save(save_path, quality=95)
    return save_name

## 远景小目标过滤 ##
# 相对小的，从大到小排序，取前nn名的平均值/2作为阈值 #
def top_N_avg(int_list,nn):
    avg_value = 0
    if len(int_list) > 0:
        int_list.sort(reverse=True)
        top_N_list = []
        # 取前NN个平均值
        NN = int(len(int_list)/2)+1 if len(int_list) < nn*2 else nn
        for i in range(0,NN):
            top_N_list.append(int_list[i])
        avg_value = np.mean(top_N_list)

    return avg_value * 1/2

def upright_to_oblique(box_upright):
    xmin, ymin, xmax, ymax = box_upright #box_upright = [xmin ,ymin ,xmax ,ymax]
    cx = (xmin + xmax)/2
    cy = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin
    angle = 0.0
    return cx,cy,w,h,angle

def montecarlo_inter(box1,box2,n=5000):
    xmin,xmax,ymin,ymax=findm([box1,box2])
    rx=[random.uniform(xmin,xmax) for i in range(n)]
    ry=[random.uniform(ymin,ymax) for i in range(n)]
    points=np.array([rx,ry]).T
    index1=np.array([ifin(points[i],box1) for i in range(n)])  ## {0,1} ##
    index2=np.array([ifin(points[i],box2) for i in range(n)])  ## {0,1} ##
    inter=np.sum(index1*index2)   ## 交集 ###
    uni=min(np.sum(index1),np.sum(index2))
    iou=inter/uni
    return round(iou,2)

def findm(boxes):
    rps=[]
    for box in boxes:
        rps.extend(rpoints(box))
    x=np.array(rps)[:,0]
    y=np.array(rps)[:,1]
    xmin=np.min(x);xmax=np.max(x);
    ymin=np.min(y);ymax=np.max(y);
    return xmin,xmax,ymin,ymax

# 计算斜框的四点坐标 ##
## rpoints入参box中的angle 是角度(r2cnn检测结果返回角度)##
def rpoints(box):
    cx,cy,w,h,angle=box[0],box[1],box[2],box[3],box[4]
    angle=angle*np.pi/180  #角度转弧度#
    p0x,p0y = rotatePoint(cx,cy, cx - w/2, cy - h/2, -angle)
    p1x,p1y = rotatePoint(cx,cy, cx + w/2, cy - h/2, -angle)
    p2x,p2y = rotatePoint(cx,cy, cx + w/2, cy + h/2, -angle)
    p3x,p3y = rotatePoint(cx,cy, cx - w/2, cy + h/2, -angle)
    points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
    return points

## 计算斜框单点坐标 ##
def rotatePoint(xc,yc, xp,yp, theta):
    xoff = xp-xc
    yoff = yp-yc
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)
    return xc+pResx,yc+pResy

def ifin(point,box):
    alpha=np.arctan2(point[1],point[0])
    #beta=-(alpha-box[4]*math.pi/180)
    beta=box[4]*np.pi/180
    rot_mat=np.array([[np.cos(beta),np.sin(beta)],[-np.sin(beta),np.cos(beta)]])
    r1=[point[0]-box[0],point[1]-box[1]]
    r0=rot_mat@r1
    if abs(r0[0])<=box[2]/2 and abs(r0[1])<=box[3]/2:
        index=1
    else: index=0
    return index

## 顺时针的坐标位置 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]
def polygon_iou(poly_points_list_1, poly_points_list_2):
    """计算任意两个凸多边形之间的 IOU"""
    #
    poly1 = Polygon(poly_points_list_1).convex_hull  # 凸多边形
    poly2 = Polygon(poly_points_list_2).convex_hull  # 凸多边形
    poly3 = poly1.intersection(poly2)
    #
    area_1 = poly1.area
    area_2 = poly2.area
    area_3 = poly3.area
    #
    iou = area_3/(area_1 + area_2 - area_3)
    return iou

## 计算交集/面积小的占比 ##
def polygon_inter(poly_points_list_1, poly_points_list_2):
    """计算任意两个凸多边形之间的 IOU"""
    #
    poly1 = Polygon(poly_points_list_1).convex_hull  # 凸多边形
    poly2 = Polygon(poly_points_list_2).convex_hull  # 凸多边形
    poly3 = poly1.intersection(poly2)
    #
    area_1 = poly1.area
    area_2 = poly2.area
    area_3 = poly3.area
    #
    iou = area_3/min(area_1,area_2)
    return round(iou,2)

# 返回坐标list #
def point_(box_upright):
    xmin, ymin, xmax, ymax = box_upright
    return [[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]]


def filter_far_small_objs(boxes,tgt_labels,nn=3):
    ## boxes : list of (label,xmin,ymin,xmax,ymax)
    seleted_boxes = []
    w_list = [(box[3]-box[1]) for box in boxes if box[0] in tgt_labels]
    h_list = [(box[4] - box[2]) for box in boxes if box[0] in tgt_labels]
    avg_w_dam = top_N_avg(w_list,nn) #宽阈值
    avg_h_dam = top_N_avg(h_list,nn) #高阈值

    for box in boxes:
        label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
        cap_w = (xmax - xmin)
        cap_h = (ymax - ymin)
        if label not in tgt_labels:
            seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))
        else:
            if cap_w > avg_w_dam and cap_h > avg_h_dam :
                seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))

    return seleted_boxes

def filter_far_small_objs_Plus(boxes,tgt_labels,nn=3):
    ## boxes : list of (label,xmin,ymin,xmax,ymax)
    seleted_boxes = []
    w_list = [(box[3]-box[1]) for box in boxes if box[0] in tgt_labels]
    h_list = [(box[4] - box[2]) for box in boxes if box[0] in tgt_labels]
    label_Detect_list =[box[5] for box in boxes if box[0] in tgt_labels]
    extra_info_list= ['K2','KG2']
    avg_w_dam = top_N_avg(w_list,nn) #宽阈值
    avg_h_dam = top_N_avg(h_list,nn) #高阈值

    for box in boxes:
        label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
        cap_w = (xmax - xmin)
        cap_h = (ymax - ymin)
        if label not in tgt_labels:
            seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))
        else:
            if len(list(set(label_Detect_list).intersection(set(extra_info_list)))) > 0 :
                if cap_w > avg_w_dam or cap_h > avg_h_dam :
                    seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))
            else:
                if cap_w > avg_w_dam and cap_h > avg_h_dam :
                    seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))

    return seleted_boxes

def filter_far_small_objs_whMax(boxes,tgt_labels,nn=3):
    seleted_boxes = []
    w_list = [max((box[3]-box[1]) ,(box[4] - box[2]) )for box in boxes if box[0] in tgt_labels]
    h_list = [min((box[3]-box[1]) ,(box[4] - box[2]) ) for box in boxes if box[0] in tgt_labels]
    avg_w_dam = top_N_avg(w_list,nn) #宽阈值
    avg_h_dam = top_N_avg(h_list,nn) #高阈值

    for box in boxes:
        label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
        cap_w = max((xmax - xmin),(ymax - ymin))
        cap_h = min((xmax - xmin),(ymax - ymin))
        if label not in tgt_labels:
            seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))
        else:
            if cap_w > avg_w_dam and cap_h > avg_h_dam :
                seleted_boxes.append((label, xmin, ymin, xmax, ymax,des,prob))
    return seleted_boxes

## 只在同类间比相对大小 ##
def filter_far_small_objs_sameLabel(boxes,tgt_labels,nn=3):
    seleted_boxes = []

    objs_label_list = [p[0] for p in boxes if p[0] in tgt_labels]
    objs_label_set = set(objs_label_list) # 所有标签
    objs_dict_byLabel = {} ## 每种标签的目标们
    ## 标签 ~ 目标们
    for label in objs_label_set:
        objs = []
        for box in boxes:
            label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
            objs.append((label, xmin, ymin, xmax, ymax,des,prob))
        objs_dict_byLabel[label] = objs

    ## 对每一种标签进行排序 ##
    for label in objs_dict_byLabel.keys():
        boxes2 = objs_dict_byLabel[label]
        w_list = [(box[3] - box[1]) for box in boxes2 if box[0] in tgt_labels]
        h_list = [(box[4] - box[2]) for box in boxes2 if box[0] in tgt_labels]
        avg_w_dam = top_N_avg(w_list, nn)  # 宽阈值
        avg_h_dam = top_N_avg(h_list, nn)  # 高阈值

        print('&&&&&&&&     label={}.  avg_w_dam={},    avg_h_dam={}   &&&&&&&&&&&&&& '.format(label,avg_w_dam,avg_h_dam))

        for box in boxes2:
            label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
            cap_w = (xmax - xmin)
            cap_h = (ymax - ymin)
            print('des={},cap_w={},cap_h={}'.format(des,cap_w,cap_h ))
            if cap_w > avg_w_dam and cap_h > avg_h_dam:
                seleted_boxes.append((label, xmin, ymin, xmax, ymax, des, prob))

    return seleted_boxes

## 比面积的相对大小 ##
def filter_far_small_objs_Area(boxes,tgt_labels,nn=3):
    seleted_boxes = []
    S_list = []
    boxes_tgt = [box for box in boxes if box[0] in tgt_labels]
    for box in boxes_tgt:
        label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
        cap_w = (xmax - xmin)
        cap_h = (ymax - ymin)
        S = cap_w * cap_h
        S_list.append(S)

    avg_S_dam = top_N_avg(S_list,nn) #面积阈值
    for box in boxes_tgt:
        label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
        cap_w = (xmax - xmin)
        cap_h = (ymax - ymin)
        S = cap_w * cap_h
        if S >avg_S_dam:
            seleted_boxes.append((label, xmin, ymin, xmax, ymax, des, prob))

    return seleted_boxes


## 比面积的绝对大小 ##
def filter_far_small_objs_absArea(boxes,tgt_labels,area_threshold):
    seleted_boxes = []
    boxes_tgt = [box for box in boxes if box[0] in tgt_labels]
    for box in boxes_tgt:
        label = box[0];xmin = box[1];ymin = box[2];xmax = box[3];ymax = box[4];des = box[5];prob = box[6]
        cap_w = (xmax - xmin)
        cap_h = (ymax - ymin)
        S = cap_w * cap_h
        # print("-----------------des={},S={}".format(des,S))
        if S > area_threshold:
            seleted_boxes.append((label, xmin, ymin, xmax, ymax, des, prob))

    return seleted_boxes

