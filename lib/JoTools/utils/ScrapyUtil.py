# -*- coding: utf-8  -*-
# -*- author: jokker -*-

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import os
import time

""""
(1) url --> html
(2) html --> bs_obj
(3) find_all 和 正则表达式 找到需要的标签
(4) .attrs['标签名'] 获取标签值
"""


class ScrapyUtil(object):

    @staticmethod
    def get_bs_obj_from_url(url):
        """从 url 获取 beautiful_soup 模板"""
        html = urlopen(url)
        bs_obj = BeautifulSoup(html, "html.parser")
        return bs_obj

    @staticmethod
    def get_bs_obj_from_url_2(url, head=None):
        """从 url 获取 beautiful_soup 模板"""
        if head is None:
            head = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.'
                              '0.3538.25 Safari/537.36 Core/1.70.3676.400 QQBrowser/10.4.3469.400',
                }
        # 可以设置表头和推出时间
        html = requests.get(url, timeout=20, headers=head)
        bs_obj = BeautifulSoup(html.text, 'html.parser')
        return bs_obj

    @staticmethod
    def load_file_from_url(url, file_save_path):
        """从 url 下载数据"""
        f = requests.get(url)
        with open(file_save_path, "wb") as file_url:
            file_url.write(f.content)

    @staticmethod
    def func_bs_obj():
        """关于 bs_obj 的一些函数整理"""
        # bs_obj.find('div', {'class': 'co_content8'}).find_all('table')  # find 和 find_all 嵌套使用
        # bs.find_all(lambda  x: 'title' in x.attrs and 'href' in x.attrs)  # find_all 和 lambda 表达式一起使用
        # each.find('a', {'href':re.compile(r'/./\d{2,10}.html')})  # find 和正则表达式一起使用
        # todo 不使用 lambda 查找存在某些关键字的标签

    # --------------------- need repair --------------------------------


if __name__ == '__main__':

    url = r'https://download.pytorch.org/whl/torch_stable.html'
    bs = ScrapyUtil.get_bs_obj_from_url(url)

    # 传入正则表达式，找到存在 title 和 href 属性的 标签
    for each in bs.find_all(lambda x: 'href' in x.attrs):  # （3）使用 find_all 正则表达式 找到所有需要的标签值
        href = each.attrs['href']
        print(href)

    # pic_path = r'https://img.18qweasd.com/d/file/html/dongman/new/2019-09-22/2f0b1407ee6b78baf090e668df11dc15.jpg'
    # save_path = r'C:\Users\Administrator\Desktop\New_frm_wprd.jpg'
    #
    # ScrapyUtil.load_file_from_url(pic_path, save_path)
