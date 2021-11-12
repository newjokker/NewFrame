import os, sys
import tensorflow as tf
import numpy as np
import cv2
import configparser
from PIL import Image
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.utils import isBoxExist
from ..detect_utils.cryption import salt, decrypt_file
from ..detect_utils.cvUtils import SegCvUtils
from ..detect_libs.abstractBase import detection

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
 
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
 
    def __init__(self, pb_path,tfconfig,log):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        graph_def = tf.GraphDef.FromString(open(pb_path, 'rb').read())#change1:input frozen_inference_graph.pb
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph,config=tfconfig)
        self.log = log
    
    def run(self, image):
        """
        Runs inference on a single image.
        需要先resize到检测需要的尺寸，然后再resize回来 
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        height = image.shape[0]
        width  = image.shape[1]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image,target_size,cv2.INTER_AREA)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [resized_image]})
        seg_map = batch_seg_map[0].astype(np.uint8)
        self.log.info(type(seg_map))
        self.log.info(seg_map.shape)
        #seg_map = Image.fromarray(seg_map.astype('uint8'))
        #seg_map_original = seg_map.resize((width,height), Image.ANTIALIAS)
        seg_map_original = cv2.resize(seg_map, (width,height))
        return resized_image, seg_map_original

class DeeplabDetection(detection,SegCvUtils):
    def __init__(self,args,objName,scriptName):
        super().__init__(objName,scriptName)
        self.readArgs(args)
        self.readCfg()
        self.log = detlog(self.modelName, self.objName, self.logID)


    def readArgs(self,args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID


    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.tfmodelName = self.cf.get(self.objName,'modelName')
        self.encryption = self.cf.getboolean("common",'encryption')
        self.debug      = self.cf.getboolean("common", 'debug')
    
 
    @try_except()
    def vis_segmentation(self,image_name,image, seg_map, h, w):
        """Visualizes input image, segmentation map and overlay view."""
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg = cv2.resize(seg_image,(w,h),cv2.INTER_AREA)
        #seg = Image.fromarray(seg_image)
        #seg = seg.resize((w,h),Image.BILINEAR)
        self.log.line('vis_segmentation')
        self.log.info(type(seg))
        if self.debug:
            if 'seg' in self.tmpPaths.keys():
                 cv2.imwrite(os.path.join(self.tmpPaths['seg'] ,image_name + '.jpg'),seg)
        return seg
 

    @try_except()
    def detect(self,im,image_name='default.jpg',color = False):
        """Inferences DeepLab model and visualizes result."""
        self.log.info('running deeplab on image %s...' % image_name)
        resized_im, seg_map = self.MODEL.run(im)
        self.log.info(seg_map.shape)
        if color:
            seg_map = self.vis_segmentation(image_name,resized_im, seg_map, im.shape[0],im.shape[1])
        else:
            seg_map = seg_map * 120
        self.log.line('after vis_segmentation')
        return seg_map

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        salt = 'txkj2019'
        bkey32 = "{: <32}".format(salt).encode("utf-8")
        modelBasePath = self.modelPath
        modelName,ext = os.path.splitext(self.tfmodelName)
        model_locked_name = modelName + '_locked' + ext
        model_src_name = model_locked_name.replace("_locked", "")
        src_Fmodel = os.path.join(self.cachePath, model_src_name)
        dst_Fmodel = os.path.join(modelBasePath, model_locked_name)
        decrypt_file(salt, dst_Fmodel, src_Fmodel)
        ### 解密后的模型 ####
        return src_Fmodel


    @try_except()
    def model_restore(self):
        objName = 'jyhdeeplab'
        if self.encryption:
            tfmodel = self.dncryptionModel()
        else:
            tfmodel = os.path.join(self.modelPath,self.tfmodelName)
        tfconfig = tf.ConfigProto()#allow_soft_placement=True)
        #tfconfig.gpu_options.allow_growth=True
        tfconfig.gpu_options.per_process_gpu_memory_fraction=self.gpuRatio
        tfconfig.gpu_options.visible_device_list=str(self.gpuID)
        self.sess = tf.Session(config=tfconfig)


        self.MODEL = DeepLabModel(tfmodel,tfconfig,self.log)
        self.warmUp()
        if self.encryption:
            modelName,ext = os.path.splitext(self.tfmodelName)
            model_locked_name = modelName + '_locked' + ext
            model_src_name = model_locked_name.replace("_locked", "")
            src_Fmodel = os.path.join(self.cachePath, model_src_name)
            self.log.info('src_Fmodel:',src_Fmodel)
            self.delete_scrfile(src_Fmodel)
            self.log.info('delete ljc dncryption model successfully! ')


    @try_except()
    def delete_scrfile(self,srcfile_path):
        os.remove(srcfile_path)
    

    @try_except()
    def warmUp(self):
        #im = Image.new('RGB', (512, 512), (255, 255, 255))
        im = 128 * np.ones((512, 512, 3), dtype=np.uint8)
        self.detect(im,'warmup.jpg')



def create_pascal_label_colormap():
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
 
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
 
    return colormap
 

def label_to_color_image(label):
    """
    Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
 
    colormap = create_pascal_label_colormap()
 
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
 
    return colormap[label]
