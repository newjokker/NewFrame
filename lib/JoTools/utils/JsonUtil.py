# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import json


class JsonUtil(object):

    @staticmethod
    def save_data_to_json_file(data, json_file_path, encoding='utf-8'):
        try:
            with open(json_file_path, 'w', encoding=encoding) as json_file:
                json.dump(data, json_file, indent=4)
            return True
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def save_data_to_json_str(data):
        return json.dumps(data)

    @staticmethod
    def load_data_from_json_file(json_file_path, encoding='utf-8'):
        try:
            with open(json_file_path, 'r', encoding=encoding) as json_file:
                return json.load(json_file)
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def load_data_from_json_str(json_str):
        return json.loads(json_str)


if __name__ == "__main__":
    a = {'name': 'jokker',
         'age': 45,
         'pice': 523,
         }

    JsonUtil.save_data_to_json_file(a, 'b.json')

    print(JsonUtil.load_data_from_json_file('b.json'))
