# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 跌倒检测
import pdb
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
import torch
import numpy as np

@DecoratorUtil.time_this
def dete_fall_ws(model_dict, data):
    try:
        model_human = model_dict["model_fall_ws_human"]
        model_pose = model_dict["model_fall_ws_pose"]
        model_deep_sort = model_dict["model_fall_ws_deep_sort"]
        model_action = model_dict["model_fall_ws_action"]

        # # dete human
        # human_dete_res = model_human.detectSOUT(path=data['path'], image=copy.deepcopy(data['im']), image_name=data['name'])
        # # dete key point
        # boxes = dete_res_to_boxes(human_dete_res)
        # point_res = model_pose.detectSOUT(copy.deepcopy(data['im']), boxes)
        # track
        #
        # human_dete_res.print_as_fzc_format()
        # print("-" * 50)
        # point_res.print_as_fzc_format()
        # print("-" * 50)
        #
        # save_path = os.path.join(r"/home/ldq/NewFrame/res", data['name'])
        # print(save_path)
        # point_res.img_path = data['path']
        # point_res.draw_res(save_path)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------





        # ---------------------- Yolo person detect ----------------------------------------------------------------
        boxes, classes, scores = model_human.detect(copy.deepcopy(data['im']))
        # 当检测为空的时候
        if not boxes:
            return

        # ---------------------- body keypoint detect --------------------------------------------------------------
        hm, cropped_boxes = model_pose.detect(copy.deepcopy(data['im']), boxes)         # heat map , 第一位是 batch size ，第二维是关键点的个数
        preds_kps, preds_scores = model_pose.postProcess(hm, cropped_boxes)
        # ------------------------ Deep-sort track -----------------------------------------------------------------
        # Adapt detections to deep sort input format
        if boxes is not None and len(boxes):
            assert boxes.shape[0] == preds_kps.shape[
                0], 'the detect box and keypoints dim not match'  # 当框数不等于点集个数的时候，直接报错

            xywhs = [bbox_rel(xyxy) for xyxy in boxes]
            xywhs = torch.Tensor(xywhs)
            tr_boxes, ids, keypoints = model_deep_sort.update(xywhs, preds_kps, preds_scores, copy.deepcopy(data['im']))

            if tr_boxes is not None and len(boxes):

                action = 'pending..'

                for tbox, tid, keypt in zip(tr_boxes, ids, keypoints):
                    # fixme 这边的意思是至少要有 30 张图片才能做动作预测，
                    if len(keypt) >= 30:
                        pts = np.array(keypt, dtype=np.float32)
                        out = model_action.predict(pts, copy.deepcopy(data['im']).shape[:2])
                        action_name = model_action.classes[out[0].argmax()]
                        print(action_name)
                        action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)

                    print(action)
        else:
            model_deep_sort.increment_ages()

        return None

    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
        print(e.__traceback__.tb_lineno)


def dete_res_to_boxes(dete_res):
    """get boxes from dete res"""
    boxes = []
    for each_dete_obj in dete_res:
        boxes.append([each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2])
    return np.array(boxes)

def bbox_rel(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


