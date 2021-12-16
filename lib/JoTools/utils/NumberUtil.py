# -*- coding: utf-8  -*-
# -*- author: jokker -*-


# 与数字有关的操作集合


class NumberUtil(object):

    @staticmethod
    def format_float(assign_float, assign_count=2):
        """float | 可以转为 float 的 str， 转为 字符串，小数点后面保存 assign_count 位"""
        if isinstance(assign_float, str):
            assign_float = float(assign_float)
        res = format(assign_float, '.{0}f'.format(max(0, assign_count)))
        return res

    @staticmethod
    def test_001():
        """中文 --> 阿拉伯数字"""
        pass

    @staticmethod
    def test_002():
        """阿拉伯数字 --> 中文"""
        pass


if __name__ == "__main__":


    print(NumberUtil.format_float("12.34"))
    print(NumberUtil.format_float(1.256445454564, 5))


