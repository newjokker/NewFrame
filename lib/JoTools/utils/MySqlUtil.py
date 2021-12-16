# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 参考 ： https://www.cnblogs.com/liubinsh/p/7568423.html

import pymysql
import logging


class MySqlUtil(object):

    def __init__(self):
        self.cursor = None
        self.db = None

    def conoect_mysql(self, host, port, user, passwd, db_name, charset='utf8'):
        """连接数据库"""
        db = pymysql.Connect(
            host=host,
            port=port,
            user=user,
            passwd=passwd,
            db=db_name,
            charset=charset,
        )
        self.db = db
        self.cursor = db.cursor()

    def close_connect(self):
        """关闭数据库连接"""
        self.cursor.close()
        self.db.close()

    @staticmethod
    def __get_select_sql_str(item_names, table_name, conditions=None):
        """得到 SQL 语句"""
        # item_names = ['Year', 'Mon', 'Day', 'Lon', 'Lat']
        # table_name = 'cimiss_surf_chn_mul_day'
        # conditions = ['month = 5', 'Day >=1', 'Day <= 10']

        if conditions is None or not conditions:
            sql_str = "select {0} from {1}".format(','.join(item_names), table_name)
        else:
            sql_str = "select {0} from {1} where {2}".format(','.join(item_names), table_name, ' and '.join(conditions))
        return sql_str

    @staticmethod
    def get_sql_str(data):
        """得到可用的字符串"""
        # FIXME 换一个名字
        if isinstance(data, int) or isinstance(data, float):
            return str(data)
        elif isinstance(data, str):
            return '"' + data + '"'
        else:
            return '"' + data + '"'
            # raise TypeError('data only can be int float or str')

    # --------------------------------------------- 增删改查 -----------------------------------------------------------
    def select_info_from_database(self, item_names, table_name, conditions):
        """从数据库获取数据"""
        if not self.cursor:
            logging.error('database havenot connect')

        # try:
        # 获取 sql 语句
        sql_str = self.__get_select_sql_str(item_names, table_name, conditions)
        # 根据 sql_str 获取需要的信息
        return self.execute_and_fetch(sql_str)

    def insert_info_to_table(self, table_name, insert_info):
        """将信息插入到数据库中去，返回是否插入成功的信息, insert_info : [{key:value}]"""
        try:
            if isinstance(insert_info, dict):  # 将传入的字典转为 列表里面的字典，格式进行统一
                insert_info = [insert_info]

            for each_item in insert_info:
                keys, valuses = zip(*each_item.items())  # 分离字典中的键和值
                valuses = map(MySqlUtil.get_sql_str, valuses)  # 将数字和字符串转为需要的格式
                sql_str = u"INSERT INTO {0} ({1}) VALUES ({2})".format(table_name, ','.join(keys), ','.join(valuses))
                print(sql_str)

                self.cursor.execute(sql_str)  # 执行
            self.db.commit()  # 提交数据，可以多次插入之后再去提交，每次插入之后提交速度会很慢
            return True
        except Exception as e:
            print(e)
            return False

    def update_info_to_table(self):
        """更新数据"""

    def delete_info_from_table(self):
        """表中删除数据"""

    # ------------------------------------------------------------------------------------------------------------------
    def execute_and_fetch(self, sql_str):
        """执行并返回值"""
        self.cursor.execute(sql_str)
        data = self.cursor.fetchall()
        return data

    def execute_and_commit(self, sql_str):
        """运行和提交"""
        self.cursor.execute(sql_str)
        self.db.commit()

    def execute(self, sql_str):
        """运行和提交"""
        self.cursor.execute(sql_str)


if __name__ == '__main__':
    # FIXME where 条件选择，可以使用关键字 in 这样比较好弄

    a = MySqlUtil()
    Host, Port, User, Passwd, Db_name = '192.168.1.236', 3306, 'root', '747225581', "world"
    # host, port, user, passwd, db_name = 'localhost', 3306, 'root','747225581', "jokker"

    a.conoect_mysql(Host, Port, User, Passwd, Db_name)

    # data_info = a.get_info_from_database(['content'], 'duanzi', ['"一" in content'])
    # # data_info = a.get_info_from_database(['*'], 'duanzi', ['content like "%鱼%"'])
    #
    # for each in data_info:
    #     print(each)

    # item_names = ['TEM_Avg', 'PRE', 'SSH']
    # table_name = 'tenfactorstatistics'
    # # conditions = ['tenorder = 15.0', 'Station_Name = "治多"']
    # conditions = ['Station_Name = "捶我"']
    #
    # data_info = a.get_info_from_database(item_names, table_name, conditions)
    #
    # for each in data_info:
    #     print(each)
