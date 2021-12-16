# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import csv


class CsvUtil(object):

    @staticmethod
    def read_csv_to_list(csv_path):
        """
        :param csv_path: str
        :return: [[1,2,3], [4,5,6], [7,8,9]]
        """
        return_data = []
        #
        with open(csv_path, encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                return_data.append(row)
        return return_data

    @staticmethod
    def read_csvs_to_dict(csv_path_dict):
        return_dict = {}
        for each_csv_path in csv_path_dict:
            try:
                return_dict[each_csv_path] = CsvUtil.read_csv_to_list(each_csv_path)
            except:
                return_dict[each_csv_path] = None
        return return_dict

    @staticmethod
    def save_list_to_csv(list_data, csv_path):
        """
        :param list_data:  [[1,2,3], [4,5,6], [7,8,9]]
        :param csv_path: str
        :return: None
        """
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(list_data)


if __name__ == '__main__':

    csvPath = r'C:\Users\Administrator\Desktop\iris2.csv'

    for each in CsvUtil.read_csv_to_list(csvPath):

        print(each)

    CsvUtil.save_list_to_csv(CsvUtil.read_csv_to_list(csvPath), r'C:\Users\Administrator\Desktop\iris2.csv')


