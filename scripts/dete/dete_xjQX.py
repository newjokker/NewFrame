# -*- coding: utf-8  -*-
# -*- author: jokker -*-

from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil
#
from JoTools.utils.DecoratorUtil import DecoratorUtil


@DecoratorUtil.time_this
def dete_xjQX(model_dict, data):
    try:
        model_xjQX_1 = model_dict["model_xjQX_1"]
        model_xjQX_2 = model_dict["model_xjQX_2"]
        #
        xjQX_dete_res = DeteRes()
        detectBoxes = model_xjQX_1.detect(data['im'], data['name'])
        results = model_xjQX_1.postProcess2(data['im'], data['name'], detectBoxes)
        #
        for xjBox in results:
            resizedName = xjBox['resizedName']
            resizedImg = im[xjBox['ymin']:xjBox['ymax'], xjBox['xmin']:xjBox['xmax']]
            segImage = model_xjQX_2.detect(resizedImg, resizedName)
            result = model_xjQX_2.postProcess(segImage, resizedName, xjBox)
            # add obj
            if "position" in result:
                x1, y1, w, h = result["position"]
                x2, y2 = x1 + w, y1 + h
                tag = result["class"]
                xjQX_dete_res.add_obj(x1, y1, x2, y2, tag, conf=-1, assign_id=-1, describe='')
        # torch.cuda.empty_cache()
        return xjQX_dete_res
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


def screen(y, img):
    # screen brightness
    _, _, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    vmedian = np.median(v)
    if vmedian < 35:
        y = '0'
    # screen obscure
    blurry = cv2.Laplacian(img, cv2.CV_64F).var()
    if blurry < 200:
        y = '0'
    return y


