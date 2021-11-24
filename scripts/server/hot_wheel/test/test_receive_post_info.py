# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import uuid
import os, sys
import argparse
from gevent import monkey
from gevent.pywsgi import WSGIServer
import datetime
import json
#
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
#

monkey.patch_all()
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/sendResult', methods=['post'])
def receive_dete_res():
    data = request.get_data()
    data = bytes.decode(data)
    data = json.loads(data)
    print(data)
    return jsonify({"status":"OK"})


def serv_start():
    global host, portNum
    http_server = WSGIServer((host, portNum), app)
    http_server.serve_forever()

def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--port', dest='port', type=int, default=3232)
    parser.add_argument('--host', dest='host', type=str, default='0.0.0.0')
    #
    parser.add_argument('--img_dir', dest='img_dir', type=str, default='/temp')
    parser.add_argument('--sign_dir', dest='sign_dir', type=str, default='/temp')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
    #
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    portNum = args.port
    host = args.host

    img_dir = args.img_dir
    sign_dir = args.sign_dir
    batch_size = args.batch_size

    dete_img_num = 0

    sign_txt_path = os.path.join(sign_dir, 'img_dir_to_dete.txt')
    if os.path.exists(sign_txt_path):
        os.remove(sign_txt_path)

    random_dir_name = str(uuid.uuid1())
    random_dir_path = os.path.join(img_dir, random_dir_name)
    os.makedirs(random_dir_path, exist_ok=True)

    url = r"http://" + host + ":" +  str(portNum) + "/receive_server"

    print(url)

    # ----------------------------------------------------------------------------------

    serv_start()














