# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
sys.path.insert(0, lib_path)

from configs import cfgs
from label_name_dict.label_dict import LABEl_NAME_MAP
NOT_DRAW_BOXES = 0
ONLY_DRAW_BOXES = -1
ONLY_DRAW_BOXES_WITH_SCORES = -2

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen', 'LightBlue', 'LightGreen'
]
FONT = ImageFont.load_default()


def draw_a_rectangel_in_img(draw_obj, box, color, width):
    '''
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    top_left, top_right = (x1, y1), (x2, y1)
    bottom_left, bottom_right = (x1, y2), (x2, y2)

    draw_obj.line(xy=[top_left, top_right],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[top_left, bottom_left],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[bottom_left, bottom_right],
                  fill=color,
                  width=width)
    draw_obj.line(xy=[top_right, bottom_right],
                  fill=color,
                  width=width)


def only_draw_scores(draw_obj, box, score, color):

    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y, x+60, y+10],
                       fill=color)
    draw_obj.text(xy=(x, y),
                  text="obj:" +str(round(score, 2)),
                  fill='black',
                  font=FONT)


def draw_label_with_scores(draw_obj, box, label, score, color):
    x, y = box[0], box[1]
    draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
                       fill=color)

    txt = LABEl_NAME_MAP[label] + ':' + str(round(score, 2))
    draw_obj.text(xy=(x, y),
                  text=txt,
                  fill='black',
                  font=FONT)


def draw_boxes_with_label_and_scores(img_array, boxes, labels, scores):

    img_array = img_array + np.array(cfgs.PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box, a_label, a_score in zip(boxes, labels, scores):

        if a_label != NOT_DRAW_BOXES:
            num_of_objs += 1
            draw_a_rectangel_in_img(draw_obj, box, color=STANDARD_COLORS[a_label], width=3)
            if a_label == ONLY_DRAW_BOXES:  # -1
                continue
            elif a_label == ONLY_DRAW_BOXES_WITH_SCORES:  # -2
                 only_draw_scores(draw_obj, box, a_score, color='White')
                 continue
            else:
                draw_label_with_scores(draw_obj, box, a_label, a_score, color='White')

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)

    return np.array(out_img_obj)


def draw_box_cv(img, boxes, labels, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        label = labels[i]
        if label != 0:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 255, 0)
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

            category = LABEl_NAME_MAP[label]

            # if scores is not None:
            #     cv2.rectangle(img,
            #                   pt1=(xmin, ymin),
            #                   pt2=(xmin+150, ymin+15),
            #                   color=color,
            #                   thickness=-1)
            #     cv2.putText(img,
            #                 text=category+": "+str(scores[i]),
            #                 org=(xmin, ymin+10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
            # else:
            #     cv2.rectangle(img,
            #                   pt1=(xmin, ymin),
            #                   pt2=(xmin + 40, ymin + 15),
            #                   color=color,
            #                   thickness=-1)
            #   cv2.putText(img,
            #                 text=category[0],
            #                 org=(xmin, ymin + 10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img

def get_rotate_box_info(img,boxes,labels,scores):
    boxes_ad=[]
    for index in range(len(boxes)):
        x_c0=boxes[index][0]
        y_c0=boxes[index][1]
        w0=boxes[index][2]
        h0=boxes[index][3]
        theta0=boxes[index][4]
        boxes_ad.append([x_c0,y_c0,w0,h0,theta0])
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)

    img_size=img.shape
    H,W=img_size[0],img_size[1]
    relation=[]
    num_of_object = 0
    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], theta0
       # if theta0>-45 :
       #    x1=int(x_c0+0.5*h0*math.sin(math.pi/180*theta0))
       #    y1=int(y_c0-0.5*h0*math.cos(math.pi/180*theta0))
       #    x2=int(x_c0-0.5*h0*math.sin(math.pi/180*theta0))
       #    y2=int(y_c0+0.5*h0*math.cos(math.pi/180*theta0))
       #    relation=[(x1/W,y1/H),(x2/W,y2/H)]
       # else:
       #    x1=int(x_c0+0.5*w0*math.cos(-math.pi/180*theta0))
       #    y1=int(y_c0-0.5*w0*math.sin(-math.pi/180*theta0))
       #    x2=int(x_c0-0.5*w0*math.cos(-math.pi/180*theta0))
       #    y2=int(y_c0+0.5*w0*math.sin(-math.pi/180*theta0))
        if theta0>-45 :
           x1=int(boxes_ad[i][0]+0.5*h0*math.sin(math.pi/180*boxes_ad[i][4]))
           y1=int(boxes_ad[i][1]-0.5*h0*math.cos(math.pi/180*boxes_ad[i][4]))
           x2=int(boxes_ad[i][0]-0.5*h0*math.sin(math.pi/180*boxes_ad[i][4]))
           y2=int(boxes_ad[i][1]+0.5*h0*math.cos(math.pi/180*boxes_ad[i][4]))
        else:
           x1=int(boxes_ad[i][0]+0.5*w0*math.cos(-math.pi/180*boxes_ad[i][4]))
           y1=int(boxes_ad[i][1]-0.5*w0*math.sin(-math.pi/180*boxes_ad[i][4]))
           x2=int(boxes_ad[i][0]-0.5*w0*math.cos(-math.pi/180*boxes_ad[i][4]))
           y2=int(boxes_ad[i][1]+0.5*w0*math.sin(-math.pi/180*boxes_ad[i][4]))
        label = labels[i]
        relation.append({'label':label,'x1':x1,'y1':y1,'x2':x2,'y2':y2})
    return relation 


def draw_rotate_box_cv(img, boxes, labels, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes_ad=[]
    for index in range(len(boxes)):
        x_c0=boxes[index][0]
        y_c0=boxes[index][1]
        w0=boxes[index][2]
        h0=boxes[index][3]
        theta0=boxes[index][4]
        boxes_ad.append([x_c0,y_c0,w0,h0,theta0])
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
   
    img_size=img.shape
    H,W=img_size[0],img_size[1]
    relation=[]
    num_of_object = 0
    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], theta0
       # if theta0>-45 :
       #    x1=int(x_c0+0.5*h0*math.sin(math.pi/180*theta0))
       #    y1=int(y_c0-0.5*h0*math.cos(math.pi/180*theta0))
       #    x2=int(x_c0-0.5*h0*math.sin(math.pi/180*theta0))
       #    y2=int(y_c0+0.5*h0*math.cos(math.pi/180*theta0))
       #    relation=[(x1/W,y1/H),(x2/W,y2/H)]
       # else:
       #    x1=int(x_c0+0.5*w0*math.cos(-math.pi/180*theta0))
       #    y1=int(y_c0-0.5*w0*math.sin(-math.pi/180*theta0))
       #    x2=int(x_c0-0.5*w0*math.cos(-math.pi/180*theta0))
       #    y2=int(y_c0+0.5*w0*math.sin(-math.pi/180*theta0))
        if theta0>-45 :
           x1=int(boxes_ad[i][0]+0.5*h0*math.sin(math.pi/180*boxes_ad[i][4]))
           y1=int(boxes_ad[i][1]-0.5*h0*math.cos(math.pi/180*boxes_ad[i][4]))
           x2=int(boxes_ad[i][0]-0.5*h0*math.sin(math.pi/180*boxes_ad[i][4]))
           y2=int(boxes_ad[i][1]+0.5*h0*math.cos(math.pi/180*boxes_ad[i][4]))
        else:
           x1=int(boxes_ad[i][0]+0.5*w0*math.cos(-math.pi/180*boxes_ad[i][4]))
           y1=int(boxes_ad[i][1]-0.5*w0*math.sin(-math.pi/180*boxes_ad[i][4]))
           x2=int(boxes_ad[i][0]-0.5*w0*math.cos(-math.pi/180*boxes_ad[i][4]))
           y2=int(boxes_ad[i][1]+0.5*w0*math.sin(-math.pi/180*boxes_ad[i][4]))
        label = labels[i]
        relation.append([x1/W,y1/H,x2/W,y2/H])
       #  print(relation)
        if label != 0:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 255, 0)
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(img, [rect], -1, color, 2)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1,4)
            category = LABEl_NAME_MAP[label]
            print((x_c,y_c),(w,h),theta)

            if scores is not None:
               print(scores)
            #     cv2.rectangle(img,
            #                   pt1=(x_c, y_c),
            #                   pt2=(x_c + 120, y_c + 15),
            #                   color=color,
            #                   thickness=-1)
            #     cv2.putText(img,
            #                 text=category+": "+str(scores[i]),
            #                 org=(x_c, y_c+10),
            #                 fontFace=1,
            #                 fontScale=1,
            #                 thickness=2,
            #                 color=(color[1], color[2], color[0]))
            # else:
            #     cv2.rectangle(img,
            #                   pt1=(x_c, y_c),
            #                   pt2=(x_c + 40, y_c + 15),
            #                   color=color,
            #                   thickness=-1)
               cv2.putText(img,
                           text=category[0],
                           org=(x_c, y_c + 10),
                           fontFace=1,
                           fontScale=1,
                           thickness=2,
                           color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img,relation


if __name__ == '__main__':
    img_array = cv2.imread("/home/yjr/PycharmProjects/FPN_TF/tools/inference_image/2.jpg")
    img_array = np.array(img_array, np.float32) - np.array(cfgs.PIXEL_MEAN)
    boxes = np.array(
        [[200, 200, 500, 500],
         [300, 300, 400, 400],
         [200, 200, 400, 400]]
    )

    # test only draw boxes
    labes = np.ones(shape=[len(boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES
    scores = np.zeros_like(labes)
    imm = draw_boxes_with_label_and_scores(img_array, boxes, labes ,scores)
    # imm = np.array(imm)

    cv2.imshow("te", imm)

    # test only draw scores
    labes = np.ones(shape=[len(boxes), ], dtype=np.float32) * ONLY_DRAW_BOXES_WITH_SCORES
    scores = np.random.rand((len(boxes))) * 10
    imm2 = draw_boxes_with_label_and_scores(img_array, boxes, labes, scores)

    cv2.imshow("te2", imm2)
    # test draw label and scores

    labels = np.arange(1, 4)
    imm3 = draw_boxes_with_label_and_scores(img_array, boxes, labels, scores)
    cv2.imshow("te3", imm3)

    cv2.waitKey(0)







