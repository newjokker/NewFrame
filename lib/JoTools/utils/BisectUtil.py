# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# 参考地址 ： https://www.cnblogs.com/skydesign/archive/2011/09/02/2163592.html
# 参考书籍 ： Python高性能编程 P62

import bisect


class BisectUtil(object):

    @staticmethod
    def find_closest(haystack, needle):
        """在列表中查找最接近目标的值，输入的数据需要是从小到大排序好的"""
        i = bisect.bisect_left(haystack, needle)
        if i == len(haystack):
            return i - 1
        elif haystack[i] == needle:
            return i
        elif i > 0:
            j = i - 1
            if haystack[i] - needle > needle - haystack[j]:
                return j
        return i


if __name__ == '__main__':

    pass
    # a = [1, 1.5, 2, 5, 6, 89, 999]
    #
    # index = BisectUtil.find_closest(a, 44)
    #
    # print(a[index])
