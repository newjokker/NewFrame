from cls_rust_data import features
import glob
import numpy as np
from PIL import Image

ss=glob.glob('/home/lsq/rusting/v0.0.2/result/*')
for s in ss:
    img=Image.open(s)
    img=np.array(img)
    print(img)
    #f=features(img,0.01)
    #print(f)