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
def dete_kkx(model_dict, data):

    try:
        model_kkxTC_1 = model_dict["model_kkxTC_1"]
        model_kkxTC_2 = model_dict["model_kkxTC_2"]
        model_kkxTC_3 = model_dict["model_kkxTC_3"]
        model_kkxQuiting = model_dict["model_kkxQuiting"]

        # kkxTC_1
        kkxTC_1_out = model_kkxTC_1.detect(data['im'], data['name'])
        if len(kkxTC_1_out[0]) > 0:
            voc_labels = model_kkxTC_1.post_process(*kkxTC_1_out)
            kkxTC_1_results = model_kkxTC_1.postProcess2(data['im'], *kkxTC_1_out)
        else:
            kkxTC_1_results = []
            #
        kkxTC_1_dete_res = DeteRes()
        kkxTC_1_dete_res.img_path = data['path']
        for i, each_res in enumerate(kkxTC_1_results):
            label, score, [xmin, ymin, xmax, ymax] = each_res
            ljc_resizedName = data['name'] + '_' + label + '_' + str(i) + '.jpg'
            # add up_right obj
            kkxTC_1_dete_res.add_obj(int(xmin), int(ymin), int(xmax), int(ymax), str(label), conf=-1, assign_id=i,
                                     describe=ljc_resizedName)
        #
        kkxTC_1_dete_res.do_nms(0.3)
        kkxTC_1_save_dir = model_kkxTC_1.resizedImgPath
        kkxTC_1_dete_res.crop_dete_obj(kkxTC_1_save_dir)
        # ---------------------------------------
        # kkxTC_2
        ###  单连接件 ###
        kkxTC_2_dete_kg_lm = kkxTC_1_dete_res.deep_copy(copy_img=False)
        kkxTC_2_dete_kg_lm.reset_alarms([])

        # 遍历每一个连接件正框
        for each_dete_obj in kkxTC_1_dete_res.alarms:
            each_dete_kg_lm = kkxTC_1_dete_res.deep_copy(copy_img=False)
            each_dete_kg_lm.reset_alarms([])
            # get array 连接件正框图片矩阵 np.array
            # each_sub_array = kkxTC_1_dete_res.get_sub_img_by_dete_obj(each_dete_obj,RGB=True)
            each_sub_array = kkxTC_1_dete_res.get_sub_img_by_dete_obj_from_crop(each_dete_obj, RGB=False)
            # 小金具定位检测结果集合 on a ljc martrix-cap
            kkxTC_2_out = model_kkxTC_2.detect(each_sub_array, data['name'])
            if len(kkxTC_2_out[0]) > 0:
                voc_labels = model_kkxTC_2.post_process(*kkxTC_2_out)
                ## 过滤最小尺寸 ##
                voc_labels = model_kkxTC_2.checkDetectBoxAreas(voc_labels)

                for each_obj in voc_labels:
                    ## label, i, xmin, ymin, xmax, ymax,p
                    new_dete_obj = DeteObj(each_obj[2], each_obj[3], each_obj[4], each_obj[5], tag=each_obj[0],
                                           conf=float(each_obj[6]), assign_id=each_dete_obj.id)
                    each_dete_kg_lm.add_obj_2(new_dete_obj)

                ## +xmap +ymap 坐标还原至原图
                each_dete_kg_lm.offset(each_dete_obj.x1, each_dete_obj.y1)
                # merge
                kkxTC_2_dete_kg_lm += each_dete_kg_lm

        # 业务逻辑：other* 和 dense内的K过滤
        kkxTC_2_dete_res = kkxTC_2_dete_kg_lm.deep_copy(copy_img=False)
        only_other_3 = kkxTC_2_dete_kg_lm.deep_copy(copy_img=False)
        only_k = kkxTC_2_dete_kg_lm.deep_copy(copy_img=False)
        only_other_3.filter_by_tags(need_tag=model_kkxTC_2.labeles_checkedOut)
        only_k.filter_by_tags(need_tag=['K'])
        kkxTC_2_dete_res.filter_by_tags(remove_tag=['K'])
        #
        for each_dete_obj in only_k:
            is_in = False
            for each_dete_obj_2 in only_other_3:
                each_iou = ResTools.polygon_iou_1(each_dete_obj.get_points(), each_dete_obj_2.get_points())  ##
                # print('--other* iou-->{} ,other*:{}, K: {}'.format(each_iou,each_dete_obj_2.get_points(),each_dete_obj.get_points()))
                if each_iou > 0.8:
                    is_in = True
            if not is_in:
                kkxTC_2_dete_res.add_obj_2(each_dete_obj)
        #
        kkxTC_2_dete_res.do_nms(0.3)
        # 删除裁剪的小图
        kkxTC_1_dete_res.del_sub_img_from_crop()

        # ---------------------------------------

        # kkxTC_3 | kkxQuiting | kkxRust
        kkxTC_dete_res = DeteRes()
        # kkxQuiting_dete_res = DeteRes()
        # kkxRust_dete_res = DeteRes()
        #
        for each_dete_obj in kkxTC_2_dete_res:
            if each_dete_obj.tag not in ['K', 'KG', 'Lm', 'K2', 'KG2']:
                continue
            #
            each_im = kkxTC_2_dete_res.get_sub_img_by_dete_obj(each_dete_obj)
            # -----------------
            # kkxTC
            label, prob = model_kkxTC_3.detect(each_im, 'resizedName')
            label = str(label)
            each_dete_obj.conf = float(prob)
            each_dete_obj.des = each_dete_obj.tag

            if label == '2' or each_dete_obj.tag == 'Lm':
                each_dete_obj.tag = 'Lm'

            elif label == '1' and prob > model_kkxTC_3.confThresh:
                each_dete_obj.tag = 'K'
            else:
                each_dete_obj.tag = 'Xnormal'
            kkxTC_dete_res.add_obj_2(each_dete_obj)
            # -----------------
            # # kkxRust
            # if "model_kkxRust" in model_dict:
            #     if "kkxRust" in model_list:
            #         new_label, conf = model_kkxRust.detect_new(each_im, name)
            #         new_dete_obj_rust = each_dete_obj.deep_copy()
            #         if new_label == 'kkx_rust' and conf > 0.8:
            #             if each_dete_obj.tag in ["Lm"]:
            #                 new_dete_obj_rust.tag = 'Lm_rust'
            #             else:
            #                 new_dete_obj_rust.tag = 'K_KG_rust'
            #         else:
            #             new_dete_obj_rust.tag = 'Lnormal'
            #         kkxRust_dete_res.add_obj_2(new_dete_obj_rust)
            # -----------------
            # kkxQuiting
            # 0:销脚可见 1:退出 2:销头销脚正对
            if each_dete_obj.tag in ["Xnormal"]:
                label, prob = model_kkxQuiting.detect(each_im, 'resizedName')
                if label == '1' and prob > 0.5:
                    new_dete_obj = each_dete_obj.deep_copy()
                    new_dete_obj.tag = 'kkxTC'
                    kkxTC_dete_res.add_obj_2(new_dete_obj)

        # torch.cuda.empty_cache()
        return kkxTC_1_dete_res
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


