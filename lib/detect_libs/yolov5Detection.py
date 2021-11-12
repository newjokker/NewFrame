#! /usr/bin/env python
import os
import cv2
import torch
import configparser
import numpy as np

from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except
from ..detect_libs.abstractBase import detection
from ..detect_utils.cryption import salt, decrypt_file

from ..yolov5_libs.models.experimental import attempt_load
from ..yolov5_libs.datasets import letterbox
from ..yolov5_libs.torch_utils import select_device
from ..yolov5_libs.general import check_img_size, non_max_suppression, scale_coords
from ..JoTools.txkjRes.deteRes import DeteRes
from ..JoTools.txkjRes.deteObj import DeteObj


class YOLOV5Detection(detection):
    def __init__(self, args, objName, scriptName):
        super(YOLOV5Detection, self).__init__(objName, scriptName)
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)
        self.device = select_device(str(self.gpuID))

    def readArgs(self, args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID
        
    @try_except()
    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.encryption = self.cf.getboolean("common", 'encryption')
        self.debug      = self.cf.getboolean("common", 'debug')
        self.model_path = self.cf.get(self.objName, 'modelName')
        self.imgsz      = self.cf.getint(self.objName, 'img_size')
        self.score      = self.cf.getfloat(self.objName, 'conf_threshold')
        self.iou        = self.cf.getfloat(self.objName, 'iou_threshold')
        self.augment    = self.cf.getboolean(self.objName, 'augment')
        self.classes    = [c.strip() for c in self.cf.get(self.objName, "classes").split(",")]
        self.class_dict = dict(zip(range(len(self.classes)), self.classes))
        self.visible_classes = [c.strip() for c in self.cf.get(self.objName, "visible_classes").split(",")]
        try:
            self.run_mode = self.cf.get('common', 'run_mode')
        except:
            self.run_mode = 'crop'

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')

        # 加密模型
        if self.encryption:
            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.model_path)
        print("self.encryption:", self.encryption)
        print(model_path)

        self.model = attempt_load(model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.half().eval()

        self.warmUp()
        self.log.info("load complete")
        print("load complete")

        # 删除解密的模型
        if self.encryption:
            os.remove(model_path)
            self.log.info('delete dncryption model successfully! ')

    @try_except()
    def warmUp(self):
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    @try_except()
    def detect(self, image, image_name="default.jpg"):
        # difference between test.py(better results) and detect.py
        h0, w0 = image.shape[:2]  # orig hw
        r = self.imgsz / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            img = image

        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32)
        img = torch.from_numpy(img).to(self.device).half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]
            det = non_max_suppression(pred, self.score, self.iou, multi_label=True,
                                      classes=[self.classes.index(c) for c in self.visible_classes])[0]

        if len(det):
            boxes = scale_coords(img.shape[2:], det[:, :4], image.shape).round().cpu().numpy()
            scores = det[:, -2].cpu().numpy()
            classes = (det[:, -1].cpu().numpy().astype(np.int8))
            self.log.info("result:")
            self.log.info(boxes)
            self.log.info(classes)
            self.log.info(scores)

            # # 清空缓存
            # torch.cuda.empty_cache()

            return boxes, classes, scores
        else:
            return [], [], []

    @try_except()
    def post_process(self, boxes, classes, scores):
        objects = []
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = self.class_dict[classes[i]]
            objects.append([label, i, int(xmin), int(ymin), int(xmax), int(ymax), float(scores[i])])
        return objects

    @try_except()
    def detectSOUT(self, path=None,image=None, image_name="default.jpg",output_type='txkj'):
        if path==None and image is None:
            raise ValueError("path and image cannot be both None")
        dete_res = DeteRes()
        dete_res.img_path = path
        dete_res.file_name = image_name
        if image is None:
            image = dete_res.get_img_array()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, classes, scores = self.detect(bgr,image_name)
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = self.class_dict[classes[i]]
            prob = float(scores[i])
            dete_obj = DeteObj(x1=int(xmin), y1=int(ymin), x2=int(xmax), y2=int(ymax), tag=label, conf=prob, assign_id=i)
            dete_res.add_obj_2(dete_obj)
        if output_type == 'txkj':
            return dete_res
        elif output_type == 'json':
            pass
        return dete_res


    @try_except()
    def dncryptionModel(self):
        if not os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        # 解密模型
        name, ext = os.path.splitext(self.model_path)
        model_origin_name = name + ext
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        # 解密后的模型
        tfmodel = os.path.join(self.cachePath, self.model_path)
        return tfmodel
