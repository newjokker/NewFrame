# -*- coding: utf-8  -*-
# -*- author: jokker -*-
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
import copy


@DecoratorUtil.time_this
def dete_fzc(model_dict, data):

    try:
        model_fzc_1 = model_dict["model_fzc_1"]
        model_fzc_2 = model_dict["model_fzc_2"]
        model_fzc_rust = model_dict["model_fzc_rust"]

        fzc_dete_res = DeteRes()
        # step_1
        dete_res_fzc = model_fzc_1.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']), image_name=data['name'])
        dete_res_fzc.do_nms()
        fzc_dete_res.filter_by_area(5000)

        # step_2

        dete_res_fzc.img_ndarry = data['im']

        for each_dete_obj in dete_res_fzc:
            crop_array = dete_res_fzc.get_sub_img_by_dete_obj_new(each_dete_obj, RGB=False, augment_parameter=[0.1, 0.1, 0.1, 0.1])
            new_label, conf = model_fzc_2.detect_new(crop_array, data['name'])
            #
            each_dete_obj.tag = new_label
            each_dete_obj.conf = conf
            #
            if each_dete_obj.tag == "fzc_broken":
                if each_dete_obj.conf > 0.9:
                    each_dete_obj.tag = "fzc_broken"
                else:
                    each_dete_obj.tag = "other_fzc_broken"
            elif each_dete_obj.tag == "other":
                each_dete_obj.tag = "other_other"
            else:
                if each_dete_obj.conf > 0.6:
                    each_dete_obj.tag = "Fnormal"
                else:
                    each_dete_obj.tag = "other_Fnormal"
            # add fzc broken
            fzc_dete_res.add_obj_2(each_dete_obj)

            # ----------------------------------------------------------------------------------------------------------

            crop_array_rust = dete_res_fzc.get_sub_img_by_dete_obj_new(each_dete_obj, RGB=False)
            rust_index, rust_f = model_fzc_rust.detect(crop_array_rust)
            rust_label = ["fzc_normal", "fzc_rust"][int(rust_index)]
            #
            each_dete_rust = each_dete_obj.deep_copy()
            each_dete_rust.tag = rust_label
            # add fzc rust
            fzc_dete_res.add_obj_2(each_dete_rust)

        # torch.cuda.empty_cache()
        return fzc_dete_res
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


