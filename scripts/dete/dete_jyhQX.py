# -*- coding: utf-8  -*-
# -*- author: jokker -*-

from lib.JoTools.txkjRes.resTools import ResTools
from lib.JoTools.utils.FileOperationUtil import FileOperationUtil
from lib.JoTools.utils.CsvUtil import CsvUtil
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.JoTools.txkjRes.deteObj import DeteObj
from lib.JoTools.txkjRes.deteAngleObj import DeteAngleObj
from lib.JoTools.utils.JsonUtil import JsonUtil
import copy

#
from JoTools.utils.DecoratorUtil import DecoratorUtil
# todo 这个函数到时候直接拷贝到这边，之前直接放在 script 文件夹中
# import judge_angle_fun

@DecoratorUtil.time_this
def dete_jyhQX(model_dict, data):

    try:

        model_jyhqx_1 = model_dict["model_jyhqx_1"]
        model_jyhqx_2 = model_dict["model_jyhqx_2"]
        model_jyhqx_3 = model_dict["model_jyhqx_3"]

        # ----------------------------------------------------------------------------------------------------------
        # dete res
        jyhqx_1_dete_res = model_jyhqx_1.detectSOUT(path=data['path'], image=data['im'], image_name=data['name'])
        jyhqx_1_dete_res.img_path = data['path']
        #

        jyhqx_1_dete_res.filter_by_tags(["fhjyz", "upring", "downring"])

        # 将 fhjyz 按照一定的规则进行扩展
        for each_obj in jyhqx_1_dete_res:
            if each_obj.tag == "fhjyz":
                each_obj.do_augment([50, 50, 50, 50], jyhqx_1_dete_res.width, jyhqx_1_dete_res.height, is_relative=False)

        # 剔除那些均压环和绝缘子无交集的部分
        dete_res_jyz = jyhqx_1_dete_res.deep_copy(copy_img=False)
        dete_res_jyz.filter_by_tags(["fhjyz"])
        #
        dete_res_jyh = jyhqx_1_dete_res.deep_copy(copy_img=False)
        dete_res_jyh.filter_by_tags(["upring", "downring"])

        # 判断绝缘子和均压环之间的交集
        for each_jyh in dete_res_jyh:
            each_dete_res_jyz = dete_res_jyz.deep_copy(copy_img=False)
            each_dete_res_jyz.filter_by_mask(each_jyh.get_points(), need_in=True, cover_index_th=0.0001)
            #
            if len(each_dete_res_jyz) < 2:
                jyhqx_1_dete_res.del_dete_obj(each_jyh)

        # 去除其中的绝缘子
        jyhqx_1_dete_res.filter_by_tags(["upring", "downring"])

        for each_obj in jyhqx_1_dete_res:
            if each_obj.tag == "downring":
                each_obj.do_augment([0, 0, 1, 0], jyhqx_1_dete_res.width, jyhqx_1_dete_res.height, is_relative=True)
            elif each_obj.tag == "upring":
                each_obj.do_augment([0, 0, 0, 1], jyhqx_1_dete_res.width, jyhqx_1_dete_res.height, is_relative=True)

        # ----------------------------------------------------------------------------------------------------------

        for each_dete_obj in jyhqx_1_dete_res:
            each_im = jyhqx_1_dete_res.get_sub_img_by_dete_obj(each_dete_obj)
            a = model_jyhqx_2.detect(each_im, name)
            #
            if len(a) < 1:
                # dete_res.del_dete_obj(each_dete_obj)
                # 在最后一步根据 des 进行过滤就行了，不用在这边删除
                pass
            else:
                point_0, point_1, point_2, point_3, _ = a[0]
                x1, y1 = int((point_0[0] + point_1[0]) / 2), int((point_0[1] + point_1[1]) / 2)
                x2, y2 = int((point_2[0] + point_3[0]) / 2), int((point_2[1] + point_3[1]) / 2)
                #
                each_dete_obj.des = "[{0},{1},{2},{3}]".format(x1, y1, x2, y2)
        # ----------------------------------------------------------------------------------------------------------

        # itoration
        for each_dete_obj in jyhqx_1_dete_res:
            # do filter
            if each_dete_obj.tag not in ['upring', 'downring'] or each_dete_obj.des in [None, "", " "]:
                continue
            # do dete
            im = jyhqx_1_dete_res.get_sub_img_by_dete_obj(each_dete_obj, RGB=False)
            # print(dete_res_3.width,dete_res_3.height)
            x_add, y_add = each_dete_obj.x1, each_dete_obj.y1
            seg_image = model_jyhqx_3.detect(im, 'resizedName')
            result = model_jyhqx_3.postProcess(seg_image, 'resizedName')

            # fixme 看一下这一步是不是需要
            if not result:
                continue

            # get_angle_1
            angle_deep_lab = result['angle']
            line_jyh = [result['xVstart'], result['yVstart'], result['xHend'], result['yHend']]
            line_jyz_1 = [result['xVstart'], result['yVstart'], result['xVend'], result['yVend']]
            # get_angle_2
            des = each_dete_obj.des
            line_jyz_2 = eval(des)
            angle_r2cnn = judge_angle_fun.angle_r2cnn(x_add, y_add, each_dete_obj, result, des)
            # judge Hnormal or fail
            Hnormal_or_fail, line_index, use_angle = judge_angle_fun.Judge_Hnormal_fail(angle_deep_lab, angle_r2cnn)
            each_dete_obj.tag = Hnormal_or_fail
            # get lines to draw
            lines_to_draw = judge_angle_fun.get_lines_to_draw(line_jyh, line_jyz_1, line_jyz_2, line_index, x_add, y_add, use_angle)
            each_dete_obj.des = str(lines_to_draw)
        # torch.cuda.empty_cache()
        return jyhqx_1_dete_res

        # ----------------------------------------------------------------------------------------------------------
    except Exception as e:
        print("error")
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)



