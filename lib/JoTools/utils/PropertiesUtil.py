# -*- coding: utf-8  -*-
# -*- author: jokker -*-


class Properties:
    """读取属性文件的类"""

    def __init__(self, file_name):
        self.file_name = file_name
        self.properties = {}
        try:
            fopen = open(self.file_name, 'r')
            for line in fopen:
                line = line.strip()
                if line.find('=') > 0 and not line.startswith('#'):
                    strs = line.split('=')
                    self.properties[strs[0].strip()] = strs[1].strip()
        except Exception as e:
            print(e)
            print("error in instantiation")
        else:
            fopen.close()

    def has_key(self, key):
        return key in self.properties

    def get(self, key, default_value=''):
        """get value by key"""
        if key in self.properties:
            return self.properties[key]
        return default_value


if __name__ == '__main__':
    p = Properties(r"D:\Code\CheckFireResult\config\test.properties")
    print(p.get("SC"))

"""
1. 文件中好像不能用于存储中文    
"""
