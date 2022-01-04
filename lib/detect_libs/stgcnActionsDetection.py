import os
import torch
import numpy as np
import configparser

from ..detect_libs.abstractBase import detection
from ..stgcnaction_libs.Models import TwoStreamSpatialTemporalGraph
from ..stgcnaction_libs.pose_utils import normalize_points_with_size, scale_pose
from ..detect_utils.log import detlog
from ..detect_utils.tryexcept import try_except,timeStamp
from ..detect_utils.cryption import salt, decrypt_file

class StgcnActionsDetection(detection):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    # def __init__(self,
    #              weight_file='./Models/TSSTG/tsstg-model.pth',
    #              device='cuda'):
    #     self.graph_args = {'strategy': 'spatial'}
    #     self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
    #                         'Stand up', 'Sit down', 'Fall Down']
    #     self.num_class = len(self.class_names)
    #     self.device = device
    #
    #     self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
    #     self.model.load_state_dict(torch.load(weight_file))
    #     self.model.eval()

    def __init__(self, args, objName, scriptName):
        super(StgcnActionsDetection, self).__init__(objName, scriptName)

        self.readArgs(args)
        self.readCfg()
        self.device = torch.device("cuda:" + str(self.gpuID)) if self.gpuID else 'cpu'
        print(f'cuda device: {self.device}')
        self.log = detlog(self.modelName, self.objName, self.logID)
        self.graph_args = {'strategy': 'spatial'}


    def readArgs(self, args):
        self.portNum = args.port
        self.gpuID = args.gpuID
        self.gpuRatio = args.gpuRatio
        self.host = args.host
        self.logID = args.logID

    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.modelName = self.cf.get('common', 'model')
        self.encryption = self.cf.getboolean("common", 'encryption')
        self.debug = self.cf.getboolean("common", 'debug')

        self.model_path = self.cf.get(self.objName, 'modelName')
        self.classes = [c.strip() for c in self.cf.get(self.objName, "classes").split(",")]
        self.num_class = len(self.classes)
        self.imgsz = self.cf.getint(self.objName, 'img_size')
        # self.inp_h = self.cf.get(self.objName, 'input_height')
        # self.inp_w = self.cf.get(self.objName, 'input_width')
        # self.pose_cfg = self.cf.get(self.objName, 'pose_cfg')
        # self.batchsize = int(self.cf.get(self.objName, 'batch_size'))
        # self.flip = self.cf.getboolean(self.objName, 'flip')

    @try_except()
    def model_restore(self):
        self.log.info('===== model restore start =====')
        # 加密模型
        if self.encryption:
            model_path = self.dncryptionModel()
        else:
            model_path = os.path.join(self.modelPath, self.model_path)
        self.log.info('model_path is:', model_path)

        # Load stgcn model
        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)

        print('Loading stgcn model from %s...' % (model_path))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # print(next(model.parameters()).device)
        #self.model.to(self.device)
        self.model.eval()

        if self.encryption:
            self.delDncryptionModel()

        self.warmUp()
        self.log.info('===== model restore end =====')
        return

    @try_except()
    def dncryptionModel(self):
        if False == os.path.exists(self.cachePath):
            os.makedirs(self.cachePath)
        ### 解密模型 ####
        name, ext = os.path.splitext(self.model_path)
        model_origin_name = name + ext
        model_locked_name = name + "_locked" + ext
        origin_Fmodel = os.path.join(self.cachePath, model_origin_name)
        locked_Fmodel = os.path.join(self.modelPath, model_locked_name)
        decrypt_file(salt, locked_Fmodel, origin_Fmodel)

        ### 解密后的模型 ####
        tfmodel = os.path.join(self.cachePath, self.model_path)
        return tfmodel

    @try_except()
    def delDncryptionModel(self):
        # 删除解密的模型
        # os.remove(self.model_path)
        # self.log.info('delete dncryption model successfully! ')
        return

    @try_except()
    def warmUp(self):
        img = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
        #keypoints = self.model(img)
        print(f'model warm up ended, Let\'s start!')
        return

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]
        
        
        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        
        
        mot = mot.to(self.device)
        pts = pts.to(self.device)
        
    
        
        out = self.model((pts, mot))
        return out.detach().cpu().numpy()
