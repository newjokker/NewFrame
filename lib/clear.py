import os
import shutil


def list_all(path):
    for i in os.listdir(path):
        r_path = os.path.join(path, i)
        if os.path.isdir(r_path):
            if i in ['.idea', '__pycache__']:
                print(r_path)
                shutil.rmtree(r_path)
 
            else:
                list_all(r_path)


dir_path = './'
list_all(dir_path)


