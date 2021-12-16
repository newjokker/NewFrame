# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import struct

# fixme 目前只支持，int, float, str，输入结构体的信息 ((int, 20), (float, 512), (char, 20)) 自动解析并返回结果，存储同样的是用这么一个结构，反向操作
# fixme 如何处理那些重复的数据结构，一直是重复的 struct，增加一个参数，询问是否重复进行读取
# fixme 目前不支持中文，不知道如何支持中文转为二进制
# fixme 使用 struct.unpack 函数解析二进制数据的时候，同时解析发多个格式的数据可能会报错，所以每次解析一个类型，比如 struct.unpack('iiffcciiffcciicciif', f.read(60))

class BytesUtil(object):

    @staticmethod
    def read_from_bytes_file(file_path, assign_struct:list, is_loop=False):
        """从二进制文件中读取需要的信息"""
        # assign_struct, (('i', 20), ('f', 512), ('c', 20))

        # 读取信息
        res = []
        read_end = False
        with open(file_path, 'rb') as f:
            #
            while True:
                for each_part in assign_struct:
                    # --------------------------------------------------
                    # print(each_part)
                    struct_str = each_part[0] * each_part[1]
                    if each_part[0] == 'f':
                        read_byte_length = each_part[1] * 4
                    elif each_part[0] == 'c':
                        read_byte_length = each_part[1]
                    elif each_part[0] == 'i':
                        read_byte_length = each_part[1] * 4
                    else:
                        raise TypeError("support f | c | i")
                    # --------------------------------------------------
                    each_struct = f.read(read_byte_length)
                    # 退出循环
                    if not each_struct:
                        read_end = True
                        break
                    # --------------------------------------------------
                    # 获取结构体信息
                    each_res = struct.unpack(struct_str, each_struct)
                    # 对 char 信息进行特殊处理
                    if each_part[0] == 'c':
                        each_res = ''.join(map(lambda x: x.decode('utf-8'), each_res))
                    res.append(each_res)

                # 是否循环读取
                if not is_loop:
                    break
                # 是否读取到文件尾部
                if read_end:
                    break

        return res

    @staticmethod
    def write_to_bytes_file(file_path, struct_info):
        """将信息写为 byte 文件，返回信息的 struct_str,用于清晰地获取信息"""
        struct_str_list = []
        with open(file_path, 'wb') as f:
            for each in struct_info:
                # fixme 这边如何处理中文和英文，因为可以转为 char，中文转为什么类型
                if isinstance(each[0], str):
                    for each_s in each:
                        f.write(bytes(each_s, encoding = "utf8"))
                    struct_str_list.append(('c', len(each)))
                elif isinstance(each[0], float):
                    for each_f in each:
                        f.write(struct.pack("f", each_f))
                    struct_str_list.append(('f', len(each)))
                elif isinstance(each[0], int):
                    for each_i in each:
                        f.write(struct.pack("i", each_i))
                    struct_str_list.append(("i", len(each)))
        return struct_str_list


if __name__ == "__main__":

    # fixme 存储完 str 之后再去存储 int or float 就会出现问题 (没发现有问题，不知道如何复现)

    file_path = r"C:\Users\14271\Desktop\del\human_face.dat"

    # a = BytesUtil.write_to_bytes_file(file_path, [[1, 2, 3, 4], [1.1, 2.2, 3.3, 4.4], 'jokker', '18761609908', [1, 2, 3, 4]])
    # a = BytesUtil.read_from_bytes_file(file_path, [('i', 4), ('f', 4), ('c', 6), ('c', 11), ('i', 4), ('c', 7)], is_loop=False)

    a = BytesUtil.write_to_bytes_file(file_path, ["start", [1, 2, 3, 4, 5], [1.2, 3.4, 5.6, 7.8, 9.0],  "end", "start", [1, 2, 3, 4, 5], [1.2, 3.4, 5.6, 7.8, 9.0],  "end"])

    print(a)

    a = BytesUtil.read_from_bytes_file(file_path, [('c', 5), ('i', 5), ('f', 5), ('c', 3)], is_loop=True)

    #
    for each in a:
        print(each)

    # todo 把获取的 xml 使用现在的格式进行存储






