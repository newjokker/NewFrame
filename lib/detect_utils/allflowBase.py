# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os, sys
import numpy as np
import time
import argparse
import configparser
import prettytable as pt
import subprocess
import json
import socket
import random
from lib.detect_utils.log import detlog
from lib.JoTools.txkjRes.deteRes import DeteRes
from lib.detect_utils.tryexcept import try_except, timeStamp
import requests


# todo 非常严格的格式检查，只要能过了格式检查，后面的错误应该都有代码可查
# todo 完善错误原因和代码自检的对应关系


class AllflowBase():

    def __init__(self, workdir):
        self.color_dict = {}
        self.config_info = {}                               
        self.all_flow = None
        # -------
        self.workdir = workdir
        self.config_path = os.path.join(self.workdir, "config.ini")
        self.log_dir = os.path.join(self.workdir, 'logs')
        self.temp_dir = os.path.join(self.workdir, 'tmpfiles')
        os.chdir(self.workdir)
        # -------
        self.host = None
        self.port = None
        self.gpu_id = None
        self.gpu_consumption = {}                                    # todo 这个应该在配置文件中写清楚，每一个 model 要多少 gpu ratio
        #
        self.parse_args()
        #
        self.read_config_from_local_file()
        #
        self.log = detlog(self.config_info["model_name"], 'allFlow', self.port)
        #
        self.pid_list = []
        #
        self.model_gpu_use = {}     # 模型对 gpu 的消耗量有 ratio 和 absolute 两个选项分别代表使用比例和绝对使用量

    @staticmethod
    def _get_color_dict_from_str(assign_color_str):
        """从配置文件中解析颜色字典"""
        assign_color_str = assign_color_str.strip()
        assign_color_dict = {}
        for each_color in assign_color_str.split(";"):
            each_color = each_color.strip()
            assign_obj_name = each_color.split(":")[0].strip()
            color_list = eval(each_color.split(":")[1].strip())
            assign_color_dict[assign_obj_name] = color_list
        return assign_color_dict

    def _parse_track(self, track):
        """读取 config 中的 track 中的内容"""
        # todo 需要进行简化，按照我自己的想法写一下（最后）
        flow = []
        #
        for step_script in track.keys():
            models = track[step_script].split(',')
            scripts = step_script.split(',')
            if len(models) != len(scripts):
                raise TypeError(" * track type error")
            plans = []
            for i in range(len(models)):
                model, input_str = models[i].split('(')
                inputData = input_str[:-1].split(':')[0]
                script = scripts[i]
                each_input_list = []
                for each_input in inputData.split(";"):
                    if each_input:
                        each_input_list.append(each_input)
                plan = {'script': script, 'model': model, 'input': each_input_list}
                plans.append(plan)
            flow.append(plans)
        self.all_flow = flow

    def _parse_gpu_consumption(self, gpu_consumption):
        """获取每个模型对 GPU 的消耗量"""
        for each_model in gpu_consumption.keys():
            gpu_ratio, gpu_abs = gpu_consumption[each_model].split('|')
            gpu_ratio = float(gpu_ratio.strip())
            gpu_abs = float(gpu_abs.strip())
            self.gpu_consumption[each_model] = {"ratio": gpu_ratio, "absolute": gpu_abs}

    def _handler(self, signal_num, frame):
        """delete sub model"""
        # fixme frame 没什么用，为了凑参数
        for res in self.pid_list:
            res.terminate()
        sys.exit(signal_num)

    @staticmethod
    def _get_free_port(ignore_port_list=None):
        """获取指定数目空闲 port"""

        if ignore_port_list is None:
            ignore_port_list = []

        port_range_tuple = (10000, 20000)
        random.seed()
        while True:
            port = random.randint(port_range_tuple[0], port_range_tuple[1])

            if port in ignore_port_list:
                continue

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = s.connect_ex(('127.0.0.1', port))
            if result:
                return port
            else:
                pass

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
        parser.add_argument('--gpuID', dest='gpuID', type=str, default='1')
        parser.add_argument('--port', dest='port', type=int, default=11223)
        parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
        parser.add_argument('--gpuRatio', dest='gpuRatio', type=float, default=0.5)
        parser.add_argument('--gpuMemory', dest='gpuMemory', type=str, default='')
        args = parser.parse_args()
        #
        self.host = args.host
        self.port = args.port
        self.gpu_id = args.gpuID

    def send_msg(self, sub_step_info, data, res):
        """send message to sub model"""
        headers = {'Connection': 'close'}
        url = r"http://{0}:{1}/{2}/{3}".format(self.host, sub_step_info['port'], self.config_info['model_name'], sub_step_info['model'])
        print("url : {0}".format(url))

        for each_input in sub_step_info['input']:
            data[each_input] = res[each_input]

        rsp = requests.post(url=url, data=json.dumps(data), headers=headers)
        #
        if rsp.status_code == 200:
            objects = json.loads(rsp.text)
            return objects, 200
        elif rsp.status_code == 207:
            return [], 207
        else:
            return None, rsp.status_code

    def detection(self, oriname, start, img_path=None):
        """单张图像检测"""
        name, extension = os.path.splitext(oriname)
        data = {"filename": name, 'path': img_path}
        #
        res, each_res, each_sub_step = {}, {}, {}
        for each_step in all_flow.all_flow:
            for each_sub_step in each_step:
                each_res, status_code = self.send_msg(each_sub_step, data, res)
                #
                if status_code in [400, 404, 500]:
                    return {}, status_code
                elif status_code in [207]:
                    res_207 = DeteRes()
                    res = res_207.get_result_construction(model_name=self.config_info['model_name'], start_time=start)
                    return res, 200
                else:
                    res.update(each_res)
        # get dete res
        dete_res = DeteRes(json_dict=each_res[each_sub_step['script']][each_sub_step['model']])
        dete_res.print_as_fzc_format()
        res = dete_res.get_result_construction(model_name=self.config_info['model_name'], start_time=start)
        # if debug save xml and jpg
        if self.config_info['test_mode']:
            # merge
            save_xml_dir = os.path.join(self.workdir, 'merge')
            os.makedirs(save_xml_dir, exist_ok=True)
            # create dir
            save_img_dir = os.path.join(self.workdir, 'result')
            os.makedirs(save_img_dir, exist_ok=True)
            # save xml | img
            dete_res.save_to_xml(os.path.join(save_xml_dir, name) + '.xml')
            dete_res.draw_dete_res(os.path.join(save_img_dir, name + '.jpg'), color_dict=self.color_dict)
        return res, 200

    def read_config_from_local_file(self):
        """从本地文件读取 config 信息"""

        cf = configparser.ConfigParser()
        cf.read(self.config_path, encoding='utf-8')

        self.check_format(cf)

        # modelName
        self.config_info['model_name'] = cf.get('common', 'model')

        # run_mode
        self.config_info['run_mode'] = cf.get('common', 'run_mode')

        # test_mode
        self.config_info['test_mode'] = cf.getboolean('common', 'debug')

        # encryption
        self.config_info['encryption'] = cf.getboolean('common', 'encryption')

        # version
        if cf.has_option('version', 'version'):
            version = cf.get('version', 'version')
        else:
            version = "0.0.0"
        self.config_info['version'] = version

        # color dict
        if cf.has_option('common', 'color_dict'):
            color_str = cf.get('common', 'color_dict')
            color_dict = self._get_color_dict_from_str(color_str)
        else:
            color_dict = {}
        self.color_dict = color_dict

        # read flow info
        self._parse_track(cf['track'])

        # read model gpu consumption from config
        self._parse_gpu_consumption(cf['gpu_consumption'])

        # print
        self.print_config_info()

    def print_config_info(self):
        """打印配置信息"""
        # --------------------------------------------------------------------------------------------------------------
        tb_model_info = pt.PrettyTable()
        # todo id, 等待检测的图片数量，端口，使用的 gpu_id, 消耗的 gpu 资源
        tb_model_info.field_names = ["  ", "status"]
        tb_model_info.add_row(["model_name", self.config_info["model_name"]])
        tb_model_info.add_row(["vision", self.config_info["version"]])
        tb_model_info.add_row(["debug", self.config_info["test_mode"]])
        tb_model_info.add_row(["encryption", self.config_info["encryption"]])
        tb_model_info.add_row(["run mode", self.config_info["run_mode"]])
        # tb.add_row(["color dict", str(self.color_dict)])
        print(tb_model_info)
        # --------------------------------------------------------------------------------------------------------------
        tb_track_info = pt.PrettyTable()
        tb_track_info.field_names = ["flow index", "script", "model", "input", "gpu use"]
        #
        for step_index, each_1 in enumerate(self.all_flow):
            for each_2 in each_1:
                script = each_2['script']
                model = each_2['model']
                input = each_2['input']
                step_index_str = "step_{0}".format(step_index)
                # fixme 这边 key 读取后就变成了小写，不知道是什么原因，是否需要进行解决？
                gpu_ratio = self.gpu_consumption[model.lower()]['ratio']
                gpu_abs = self.gpu_consumption[model.lower()]['absolute']
                tb_track_info.add_row([step_index_str, script, model, ",".join(input), "{0} | {1}".format(gpu_ratio, gpu_abs)])
        print(tb_track_info)

    def start_sub_model(self):
        """执行 cmd 操作"""
        port_used = []
        for each_step_info in self.all_flow:
            for each_sub_step in each_step_info:
                obj_name = each_sub_step['model']
                script = each_sub_step['script']
                each_port = self._get_free_port(port_used)
                port_used.append(each_port)
                each_sub_step['port'] = each_port
                bug_file = open(os.path.join(self.log_dir, "bug_" + obj_name + ".txt"), "w+")
                cmd_str = "python3 scripts/{0}.py --host {1} --port {2} --gpuID {3} --gpuRatio {4} --logID {5} --objName {6}".format(
                    script, self.host, each_port, self.gpu_id, self.gpu_consumption[obj_name.lower()]['ratio'], self.port, obj_name)
                print(cmd_str)
                each_pid = subprocess.Popen(cmd_str.split(), stdout=None, stderr=bug_file, shell=False)
                each_sub_step['pid'] = each_pid
                self.pid_list.append(each_pid)

    def check_format(self, cf):
        """格式检查"""

        print("* 准备对格式进行严格的检查")

        # fixme 严格的格式检查，



this_dir = os.path.dirname(os.path.abspath(__file__))
# work_dir/lib/this_dir
work_dir = os.path.dirname(os.path.dirname(this_dir))
# get all_flow
all_flow = AllflowBase(work_dir)













