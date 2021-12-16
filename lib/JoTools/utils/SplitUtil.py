# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import copy
import math


class SplitUtil(object):

    @staticmethod
    def split_by_every_count(data_list, assign_count):
        """按照每一份有多少个进行分组，除了最后一份，每一份的个数相同，可以选着是否保留最后一份"""
        # init
        res = []
        index = 0
        each_data_list = []
        # split
        for each_data in data_list:
            if index < assign_count:
                index += 1
                each_data_list.append(each_data)
            else:
                index = 0
                res.append(copy.deepcopy(each_data_list))
                each_data_list = []
        # rest
        if index > 0:
            res.append(copy.deepcopy(each_data_list))
        return res

    @staticmethod
    def split_to_assign_part(data_list, assign_part_num):
        """分成指定的份数，每一份的个数不一定相同"""
        # init
        res = []
        for i in range(assign_part_num):
            res.append([])
        # split
        for each_index, each_data in enumerate(data_list):
            res[each_index % assign_part_num].append(each_data)
        return res



if __name__ == "__main__":


    # test_data = list(range(200))
    #
    # # res = SplitUtil.split_to_assign_part(test_data, 7)
    # res = SplitUtil.split_by_every_count(test_data, 5)
    #
    # for each in res:
    #     print(each)



    test_data = []
    with open(r"C:\Users\14271\Desktop\fzc_step_one_20K\out_fzc_step_one_0.txt", 'r') as txt_file:
        for each_line in txt_file:
            each_line = each_line.strip()
            test_data.append(float(each_line.split(' --> ')[1][:-3]))

    res = SplitUtil.split_by_every_count(test_data, assign_count=10)

    for each in res:
        print(sum(each)/len(each))