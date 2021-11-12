# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:28:39 2020

@author: admin
"""

import sys

import os,glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import pandas as pd
from skimage.feature import texture
from skimage.measure import shannon_entropy
from sklearn.metrics.cluster import entropy
from sklearn.cluster import KMeans
from sklearn import preprocessing 

import warnings
  
def fetch_mask(img):
    #img=Image.fromarray(img)
    h,s,v=img.convert('HSV').split()
    [h,s,v]=list(map(np.array,[h,s,v]))
    mask=np.logical_or((h<29/360*255),(h>320/360*255))*np.logical_and((s>7*2.55),(s<65*2.55))*\
            np.logical_and((v>16*2.55),(v<62*2.55))
    return mask
    
def area(mask):
    mask=np.array(mask,dtype=int)
    (w,h)=np.shape(mask)
    a=np.sum(mask)/(w*h)
    gcm=texture.greycomatrix(mask,[1],[0,np.pi/2],levels=2)
    s=np.mean(texture.greycoprops(gcm,'correlation'))
    return a,s

def vectors(img):
    img=Image.fromarray(np.uint8(img))
    h,s,v =img.convert('HSV').split()
    h,s,v=map(np.array,[h,s,v])
    tg=[h,s,v]
    tg_vec=list(map(np.median,tg))
    return tg_vec

def coarse(img,mask_m):
    img=Image.fromarray(np.uint8(img))
    
    image=np.array(img.convert('L'))*mask_m
    gcm_1=texture.greycomatrix(image,[1],[0,np.pi/2],
                             normed=True,symmetric=True,levels=256)
    gcm_2=texture.greycomatrix(image,[5],[0,np.pi/2],
                             normed=True,symmetric=True,levels=256)

    energy_1=np.mean(texture.greycoprops(gcm_1,'energy'))
    contrast_1=np.mean(texture.greycoprops(gcm_1,'contrast'))/(256**2)
    energy_2=np.mean(texture.greycoprops(gcm_2,'energy'))
    contrast_2=np.mean(texture.greycoprops(gcm_2,'contrast'))/(256**2)
    c=[energy_1,contrast_1,energy_2,contrast_2]
    return c
    

def relu(x):
    if x<0:x=0
    return x   
 
def anchor(mask,sx,sy):   
    block_sum=np.add.reduceat(np.add.reduceat(mask, np.arange(0, mask.shape[0], sy), axis=0),
                                      np.arange(0, mask.shape[1], sx), axis=1)

    anchor=np.where(block_sum==np.max(block_sum))
    y=(anchor[0][0])*sy;x=(anchor[1][0])*sx;
    return [x,y]
    
def features(img,min_area):
    mask=fetch_mask(img)
    w,h=img.size
    num=8
    if np.sum(mask)<min_area*(w*h):
        f=[np.float64(0)]*num
        return f
    s=20
    sx=round(w/s);sy=round(h/s)
    x,y=anchor(mask,sx,sy)
    a,b=area(mask)
    img_m=np.array(img)
    target=img_m[y:y+sy,x:x+sx]
    mask_m=mask[y:y+sy,x:x+sx]
    tg_vec=vectors(target)
    c=coarse(target,mask_m)
    t1,t2,t3=tg_vec
    f=[a,b,t1,t2]+c
    return f

"""
def data_label(imgs_path):
    rust=glob.glob(imgs_path+'/rust/*.jpg')
    normal=glob.glob(imgs_path+'/normal/*.jpg')
    imgs=rust+normal
    n=len(rust)
    m=len(normal)
    X=[]
    y=[]
    names=[]
    random.seed(0)
    rlist=list(range(n+m))
    random.shuffle(rlist)

    for i in rlist:
        img=Image.open(imgs[i])
        f=features(img,min_area)
        X.append(f)
        if i<n:
            y.append(1)
            
        else:
            y.append(0)
        name=os.path.split(imgs[i])[1].encode('utf-8').decode('utf-8')
        names.append(name)
    X=np.array(X); y=np.array(y)
    return X, y,names
     
def raw_data(imgs_path):
    imgs=glob.glob(imgs_path+'/*.jpg')
    n=len(imgs)
    assert n != 0, 'no picture'
    X=[];
    for i in range(n):
        img=Image.open(imgs[i])
        f=features(img,min_area)
        X.append(f)
    X=np.array(X)
    return X,imgs
def save_label(imgs_path,file):
    X,y,names=data_label(imgs_path)
    columns=['area','corr','h','s','c1','c2','c3','c4']
    df=pd.DataFrame(X,columns=columns)
    df['label']=y
    df.drop(index=df.loc[df['area']<= mini_area].index,inplace=True)
    df.to_csv(imgs_path+file,index=False)

       
if __name__ == '__main__':  
    start=time.time()
    
    imgs_path='D:/corrosion/train'
    file='/data_train.csv'
    save_label(imgs_path,file)
    imgs_path='D:/corrosion/eval'
    file='/data_eval.csv'
    save_label(imgs_path,file)
    print('written')
    
    end=time.time()
    print('consuming time:{:.2f}s'.format(end-start))
""" 
