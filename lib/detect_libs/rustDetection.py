import cv2
import numpy as np
import time
import for_crop_rotation
import joblib,os
import abc,configparser
from PIL import Image
from .abstractBase import detection
from ..detect_utils.log import detlog
from ..rust_cls_libs import cls_rust_data
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.cryption import decrypt_file, salt
import segment


class rustDetection(detection):
    def __init__(self,args,objName,scriptName):
        super(rustDetection, self).__init__(objName,scriptName)
        
        self.readCfg()
        self.args = args
        self.readArgs()
        self.log = detlog(self.modelName, self.objName, self.logID)
        #self.WORKDIR=
        
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
        self.encryption = self.cf.getboolean("common", 'encryption')

        self.tfmodelName = self.cf.get(self.objName,'modelname')
        self.min_area = eval(self.cf.get(self.objName,'min_area'))       
                           
    @try_except()
    def model_restore(self):   
        self.log.info('===== model restore start =====')

        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            self.log.info(os.path.join(self.modelPath,self.tfmodelName))
            tfmodel = os.path.join(self.modelPath,self.tfmodelName)
        if not os.path.isfile(tfmodel):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel))

        self.model=joblib.load(tfmodel)
        if self.encryption:
            os.remove(tfmodel)
        self.log.info('===== model restore end =====')


    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)

        ### 解密模型 ####
        name, ext = os.path.splitext(self.tfmodelName)
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, self.tfmodelName)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        return origin_Fmodel


    @try_except()
    def warmUp(self):
        pass
        
    @abc.abstractmethod
    def detect(self, im,boxes,image_name,box_type,cats):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        if im is None:
            self.log.info("【Warning】:"+ image_name +"  == None !!!")
            return []
         
        y=self.predict(im,boxes,box_type,cats)  
        y=list(map(int,y))
        return y
  
    @abc.abstractmethod
    def createTmpDataDir(self):
        pass
    
    @try_except()
    def predict(self,img,boxes,box_type,cats):
        self.log.info('in predict')
        self.log.info(type(self.model))
        X=[]
        if len(boxes) == 0:
            return []
        for i,box in enumerate(boxes):
   
            save_path=os.path.join(self.WORKDIR,'tmpfiles/rust_cls_demo')
            save_file=os.path.join(save_path,str(i)+'.jpg')
            crop=self.save_read(save_file,box,img,box_type)
            crop=self.cut(crop,cats[i])
            f=cls_rust_data.features(crop,self.min_area)
            
            X.append(f)
        self.log.info(type(f))
        self.log.info(type(self.model))        
        y=self.model.predict(X)
        pred_y=self.amend(X,y)
        return pred_y
     
    @try_except()
    def amend(self,X,pred_y):
        X=np.array(X)
        n=np.size(pred_y)
        for i in range(n):
            if X[i,0]<=self.min_area:
                pred_y[i]=0
        return pred_y
    
    @try_except()    
    def save_read(self,save_file,box,img,box_type):
        if box_type=='upright':
            [xmin,xmax,ymin,ymax]=box
            crop=img[ymin:ymax,xmin:xmax]
            cv2.imwrite(save_file,crop,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            crop=Image.open(save_file)
        elif box_type=='rotated':
            [xc,yc,w,h,theta]=box
            rect=for_crop_rotation.pack(box)
            crop,_=for_crop_rotation.crop_rect(img,rect)
            cv2.imwrite(save_file,crop,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            crop=Image.open(save_file)
        else:
            raise ValueError
        os.remove(save_file)
        return crop
            
    @try_except()
    def cut(self, img, cat):
        
        #if cat in ['PGuaBan', 'UGuaHuan',  'ZGuaBan']:
        if cat in [1,2,3,5]:
            img=segment.cut_bg_sconnect(img)
        #elif cat in ['Sanjiaoban']:
        elif cat in [4]:
            img=segment.cut_bg_tri(img)
        return img
    
        
    
    