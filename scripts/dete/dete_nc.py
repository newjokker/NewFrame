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
def dete_nc(model_dict, data):
    try:
        model_nc = model_dict["model_nc"]
        nc_dete_res = model_nc.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']), image_name=data['name'])
        # torch.cuda.empty_cache()
        return nc_dete_res
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)
