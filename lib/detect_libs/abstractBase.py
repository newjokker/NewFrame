import abc
import os
from ..detect_utils.utils import check_record_file_path
import uuid

class detection(object):
    def __init__(self, objName, scriptName):
        self.WORKDIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.cfgPath   = os.path.join(self.WORKDIR, "config.ini")
        self.modelPath = os.path.join(self.WORKDIR, "models")
        self.tmpPath   = os.path.join(self.WORKDIR, "tmpfiles")
        self.scriptTmpPath = os.path.join(self.tmpPath, scriptName)
        # 每个生成不一样的
        self.cachePath = os.path.join(os.path.expanduser('~'), '.cache')
        self.cachePath = os.path.join(self.cachePath, str(uuid.uuid1()))
        os.makedirs(self.cachePath, exist_ok=True)
        #os.makedirs(self., exist_ok=True)
        
        self.objName = objName
        self.scriptName = scriptName
        self.tmpPaths = {}
        check_record_file_path(self.scriptTmpPath)

    def getTmpPath(self, name, scriptName=None):
        if scriptName != None:
            tmpPath = os.path.join(self.tmpPath, scriptName, name)
        else:
            tmpPath = os.path.join(self.scriptTmpPath, name)
        check_record_file_path(tmpPath)
        self.tmpPaths[name] = tmpPath
        return tmpPath

    @abc.abstractmethod
    def readArgs(self, args):
        """
        读取通过args传入的参数
        """
        pass

    @abc.abstractmethod
    def readCfg(self):
        """
        读取配置文件config.ini中的信息
        """
        pass

    @abc.abstractmethod
    def model_restore(self):
        """
        预加载模型
        """
        pass

    @abc.abstractmethod
    def warmUp(self):
        """
        调用检测函数进行热身
        """
        pass

    @abc.abstractmethod
    def detect(self, im, image_name="default.jpg"):
        """
        检测图片
        """
        pass
  
    @abc.abstractmethod
    def createTmpDataDir(self):
        """
        创建临时文件文件夹
        """
        pass

