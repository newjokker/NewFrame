# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import pickle


class PickleUtil(object):

    @staticmethod
    def load_data_from_pickle_file(pickle_file_path):
        with open(pickle_file_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    @staticmethod
    def save_data_to_pickle_file(data, pickle_file_path):
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)


if __name__ == '__main__':

    a = [1, 2, 3, 4, 5, 6]

    b = [1, '2', a]

    PickleUtil.save_data_to_pickle_file(b, 'a.pkl')

    print(PickleUtil.load_data_from_pickle_file('a.pkl'))
