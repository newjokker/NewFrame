import os
import torch
from torchvision import transforms as T
import cv2,copy
import json
import math
import os, sys
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..','r2cnnPytorch_libs')
sys.path.insert(0, lib_path)
import configparser
import numpy as np
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_libs.abstractBase import detection

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg

from ..JoTools.txkjRes.deteRes import DeteRes
from ..JoTools.txkjRes.deteAngleObj import DeteAngleObj

from Crypto.Cipher import AES
import struct
from xml.etree.ElementTree import Element,ElementTree
from xml.etree.ElementTree import tostring
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET


class R2cnnDetection(detection):
    def __init__(self,args,objName,scriptName):
        super(R2cnnDetection, self).__init__(objName,scriptName)
        self.suffixs = ['.pth', '.pt', '.meta', '.index', '.data-00000-of-00001']
        self.args = args
        self.scriptName = scriptName

        self.readArgs()
        self.readCfg()

        self.log = detlog(self.modelName, self.objName, self.logID)

        self.setCfg()

        self.save_dir = os.path.join(self.WORKDIR, "merge")

    def readArgs(self):
        self.portNum = self.args.port
        self.gpuID = self.args.gpuID
        self.gpuRatio = self.args.gpuRatio
        self.host = self.args.host
        self.logID = self.args.logID

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.debug = self.cf.getboolean('common','debug')
        self.modelName = self.cf.get('common', 'model')
        self.CATEGORIES =[c.strip() for c in self.cf.get(self.objName, "classes").split(",")]
        self.VISIBLE_CLASSES = tuple(self.cf.get(self.objName,'visible_classes').strip(',').split(','))
        self.tfmodelName = self.cf.get(self.objName,'modelName')
        self.encryption = self.cf.getboolean("common",'encryption')
        config_file = os.path.join(lib_path,'configs/e2e_r2cnn_R_50_FPN_1x.yaml')
        cfg.merge_from_file(config_file)
        self.confidence_threshold=self.cf.getfloat(self.objName, 'confidence_threshold')
        self.min_image_size=self.cf.getint(self.objName, 'min_image_size')
        #        
        cfg.MODEL.RETINANET.NUM_CLASSES = self.cf.getint(self.objName, 'num_class')
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = self.cf.getint(self.objName, 'num_class')


    @try_except()
    def setCfg(self):
        self.log.info(self.gpuID)
        self.cfg = cfg.clone()

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        name,ext = os.path.splitext(self.tfmodelName)
        self.log.info(name)
        self.log.info(ext)

        model_origin_name = name + ext
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        # for sf in self.suffixs:
        #     model_origin_name = name + ext + sf
        #     model_locked_name = name + "_locked" + ext + sf
        #     origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        #     locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        #     decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        ### 解密后的模型 ####
        self.log.info('in dncryptionModel')
        self.log.info(self.tfmodelName)
        #tfmodel = os.path.join(self.cachePath, self.tfmodelName).replace('_locked','')
        tfmodel = os.path.join(self.cachePath, self.tfmodelName)
        return tfmodel

    @try_except()
    def model_restore(self):
        self.log.info('model restore start')
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuID)
        torch.cuda.set_device(self.gpuID) 
        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath,self.tfmodelName)
        self.log.info('tfmodel')

        # if not os.path.isfile(tfmodel + '.meta'):
        #     raise IOError(('{:s} not found.\nDid you download the proper networks from '
        #                    'our server and place them properly?').format(tfmodel + '.meta'))
        self.model = build_detection_model(cfg)
        #with torch.no_grad():
        self.model.eval()
        self.device = torch.device('cuda:' + str(self.gpuID))
        self.model.to(self.device)
        save_dir = self.modelPath
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        weights = self.tfmodelName
        _ = checkpointer.load(tfmodel)
        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")


        # load network
        #self.warmUp()
        self.log.info('model restore end')
        if self.encryption:
            self.delDncryptionModel()
        return

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    @try_except()
    def delDncryptionModel(self):
        ###删除模型 ####
        for sf in self.suffixs:
            model_origin_name = self.tfmodelName + sf
            origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
            if os.path.exists(origin_Fmodel):
                os.remove(origin_Fmodel)
        self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        im = 128 * np.ones((800,1200,3),dtype=np.uint8)
        v = self.detect(im, 'warmup.jpg')
        print('r2cnn warmUp done')

    @timeStamp()
    @try_except()
    def detect(self,im,image_name="default.jpg",mirror=False):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        if im is None:
            self.log.info("【Waring】:"+ image_name +"  == None !!!")
            return []
        predictions = self.compute_prediction(im)
        predictions = self.select_top_predictions(predictions)
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        quad_boxes = predictions.quad_bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        image = im.copy()

        points_list = []

        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]

        i = 0
        for quad_box, box, color in zip(quad_boxes, boxes, colors):
            box = box.to(torch.int64)
            quad_box = quad_box.to(torch.int64)
            point_1 = (int(quad_box[0]), int(quad_box[1]))
            point_2 = (int(quad_box[2]), int(quad_box[3]))
            point_3 = (int(quad_box[4]), int(quad_box[5]))
            point_4 = (int(quad_box[6]), int(quad_box[7]))
            points_list.append([point_1,point_2, point_3, point_4,labels[i]])

            #if self.debug:
            #    cv2.line(image, (quad_box[0], quad_box[1]), (quad_box[2], quad_box[3]), color, 2)
            #    cv2.line(image, (quad_box[2], quad_box[3]), (quad_box[4], quad_box[5]), color, 2)
            #    cv2.line(image, (quad_box[4], quad_box[5]), (quad_box[6], quad_box[7]), color, 2)
            #    cv2.line(image, (quad_box[6], quad_box[7]), (quad_box[0], quad_box[1]), color, 2)
            i += 1

        if self.debug:
            result_img = self.overlay_class_names(image , predictions)
            out_dir = os.path.join(self.tmpPath,self.scriptName,'r2cnn')
            os.makedirs(out_dir,exist_ok=True)
            out_path = os.path.join(out_dir, image_name+'.jpg')
            cv2.imwrite(out_path, result_img)

        return points_list  # list of 斜框连接件的四点坐标

    @try_except()
    def detectSOUT(self, path=None, image=None, image_name="default.jpg", output_type='txkj'):
        if path == None and image is None:
            raise ValueError("path and image cannot be both None")
        dete_res = DeteRes()
        dete_res.img_path = path
        dete_res.file_name = image_name
        if image is None:
            image = dete_res.get_img_array()


        predictions = self.compute_prediction(image)
        predictions = self.select_top_predictions(predictions)
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        quad_boxes = predictions.quad_bbox

        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]

        i = 0
        dete_res = DeteRes()
        for quad_box, box in zip(quad_boxes, boxes):
            box = box.to(torch.int64)
            quad_box = quad_box.to(torch.int64)

            # cal angle and center point
            cx = int((quad_box[0] + quad_box[2] + quad_box[4] + quad_box[6]) / 4)
            cy = int((quad_box[1] + quad_box[3] + quad_box[5] + quad_box[7]) / 4)

            # cal angle and w h
            dis_1 = math.sqrt((quad_box[3] - quad_box[5]) ** 2 + (quad_box[2] - quad_box[4]) ** 2)
            dis_2 = math.sqrt((quad_box[3] - quad_box[1]) ** 2 + (quad_box[2] - quad_box[0]) ** 2)

            if dis_1 > dis_2:
                if quad_box[2] == quad_box[4]:
                    angle = math.pi / 2
                else:
                    angle = math.atan((quad_box[3] - quad_box[5]) / (quad_box[2] - quad_box[4]))
            else:
                if quad_box[0] == quad_box[2]:
                    angle = math.pi / 2
                else:
                    angle = math.atan((quad_box[1] - quad_box[3]) / (quad_box[0] - quad_box[2]))

            w = max(dis_1, dis_2)
            h = min(dis_1, dis_2)
            dete_res.add_angle_obj(cx, cy, w, h, angle, labels[i], assign_id=i)

            i += 1

        return dete_res

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            #image_list = image_list.to(self.device)
            print('len:',len(image_list.tensors))
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        quad_boxes = predictions.quad_bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for quad_box, box, color in zip(quad_boxes, boxes, colors):
            box = box.to(torch.int64)
            quad_box = quad_box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            # image = cv2.rectangle(
            #     image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            # )
            cv2.line(image, (quad_box[0], quad_box[1]), (quad_box[2], quad_box[3]), color, 2)
            cv2.line(image, (quad_box[2], quad_box[3]), (quad_box[4], quad_box[5]), color, 2)
            cv2.line(image, (quad_box[4], quad_box[5]), (quad_box[6], quad_box[7]), color, 2)
            cv2.line(image, (quad_box[6], quad_box[7]), (quad_box[0], quad_box[1]), color, 2)


        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        # print("labels:",labels)
        # print("scores:",scores)
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox
        # print("labels:",labels)
        # print("boxes:",boxes)
        template = "{}: {:.2f}"
        #for box, score, label in zip(boxes, scores, labels):
        #    x, y = box[:2]
        #    s = template.format(label, score)
        #    cv2.putText(
        #        image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        #    )

        return image

    def resize_box(self,box,img,name):
    #img=Image.fromarray(img)
        #w0, h0 = img.size
        h0,w0,_=np.shape(img)
       
        xc,yc,w,h,theta=box           
        if w0<h0:
            nw=self.cfgs.IMG_SHORT_SIDE_LEN
            nh=nw/w0*h0
        else:
            nh=self.cfgs.IMG_SHORT_SIDE_LEN
            nw=nh/h0*w0     
        new_xc=xc*w0/nw;new_yc=yc*h0/nh
        new_w=w*w0/nw;new_h=h*h0/nh
        box=[new_xc,new_yc,new_w,new_h,theta]
        box=list(map(float,box))
        return box

    def resize_box_upright(self,box,img,name):
        h0,w0,_=np.shape(img)
        xmin,ymin,xmax,ymax=box
        xc=(xmax+xmin)/2;yc=(ymax+ymin)/2
        w=xmax-xmin;h=ymax-ymin
        if w0<h0:
            nw=self.cfgs.IMG_SHORT_SIDE_LEN
            nh=nw/w0*h0
        else:
            nh=self.cfgs.IMG_SHORT_SIDE_LEN
            nw=nh/h0*w0     
        new_xc=xc*w0/nw;new_yc=yc*h0/nh
        new_w=w*w0/nw;new_h=h*h0/nh   
        xmin=new_xc-new_w/2;ymin=new_yc-new_h/2;
        xmax=new_xc+new_w/2;ymax=new_yc+new_h/2
        box=[xmin,xmax,ymin,ymax]
        box=list(map(int,box))
        return box

    def minirect(self, box, h, w):
        points = self.rpoints(box)  ## 4点坐标

        xmin = np.min(points[:, 0]);
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1]);
        ymax = np.max(points[:, 1])
        rect = [xmin, xmax, ymin, ymax]
        rect = self.restrict(rect, h, w)
        return rect

    def minirect2(self, points, h, w):
        points =  np.array(points, dtype=int) ## 4点坐标

        xmin = np.min(points[:, 0]);
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1]);
        ymax = np.max(points[:, 1])
        rect = [xmin, xmax, ymin, ymax]
        rect = self.restrict(rect, h, w)
        return rect

    ## 返回四点坐标 ##
    def rpoints(self, box):
        cx, cy, w, h, angle = box
        angle = angle * math.pi / 180
        p0x, p0y = self.rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        p1x, p1y = self.rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        p2x, p2y = self.rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        p3x, p3y = self.rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
        points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
        points = np.array(points, dtype=int)
        return points

    def rotatePoint(self, xc, yc, xp, yp, theta):
        xoff = xp - xc
        yoff = yp - yc
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * xoff + sinTheta * yoff
        pResy = - sinTheta * xoff + cosTheta * yoff
        return xc + pResx, yc + pResy

    def restrict(self, rect, h, w):
        [xmin, xmax, ymin, ymax] = rect
        xmin = max(xmin, 0);
        ymin = max(ymin, 0)
        xmax = min(w - 1, xmax);
        ymax = min(h - 1, ymax)
        rect = [xmin, xmax, ymin, ymax]
        rect = list(map(int, rect))
        return rect


    
def delete_scrfile(srcfile_path):
    os.remove(srcfile_path)
