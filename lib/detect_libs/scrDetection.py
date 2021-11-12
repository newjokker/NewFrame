import math
import os,cv2
import numpy as np
from PIL import Image
# from scripts.drawbox import minirect
from .r2cnnDetection import R2cnnDetection
from ..r2cnn_libs.configs import cfgs
from ..detect_utils.tryexcept import try_except,timeStamp
from ..r2cnn_libs.box_utils import draw_box_in_img
from ..r2cnn_libs.configs import cfgs


class scrDetection(R2cnnDetection):
    
    @timeStamp()
    @try_except()
    def detect(self,im,image_name="default.jpg",mirror=False):
        self.log.info('=========================')
        self.log.info(self.modelName + ' detection start')
        if im is None:
            self.log.info("【Warning】:"+ image_name +"  == None !!!")
            return []

        resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
        det_boxes_r_, det_scores_r_, det_category_r_ = \
                self.sess.run(
                    [self.img_batch, self.det_boxes_h, self.det_scores_h, self.det_category_h,
                     self.det_boxes_r, self.det_scores_r, self.det_category_r],
                    feed_dict={self.img_plac: im}
                )
                
        rects=[]
        h,w,_=np.shape(im)
        #self.Box_type='upright'
        if self.Box_type=='upright':
            for box in det_boxes_r_:
                pass_filter=filter(self,box)
                if pass_filter:
                    rect=minirect(self.resize_box(box,im,image_name),h,w)
                    rects.append(rect)
        elif self.Box_type=='rotated':
            for box in det_boxes_r_:
                pass_filter=filter(self,box)
                if pass_filter:
                    rect=self.resize_box(box,im,image_name)
                    rects.append(rect)
        else:
            raise NameError
        cats=list(map(int,det_category_r_))
        results=[{'boxes':rects,'name':image_name,'box_type':self.Box_type,'cats':cats}]
        

        self.log.info(results)
        # if self.debug:
        #     rotate_image_path = self.getTmpPath('rotate')
        #     rotate_image,_ = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
        #                                                  boxes=det_boxes_r_,
        #                                                  labels=det_category_r_,
        #                                                  scores=det_scores_r_)
        #     save_path = os.path.join(rotate_image_path,image_name)
        #     cv2.imencode('.jpg', rotate_image)[1].tofile(save_path)
        #
        return results
    
    def resize_box(self,box,img,name):
        xc,yc,w,h,theta=box
        img=Image.fromarray(img)
        w0, h0 = img.size
        if w0<h0:
            nw=cfgs.IMG_SHORT_SIDE_LEN
            nh=nw/w0*h0
        else:
            nh=cfgs.IMG_SHORT_SIDE_LEN
            nw=nh/h0*w0     
        new_xc=xc*w0/nw;new_yc=yc*h0/nh
        new_w=w*w0/nw;new_h=h*h0/nh
        box=[new_xc,new_yc,new_w,new_h,theta]
        box=list(map(float,box))
        return box
        
    def minirect(self,box):
        points=self.rpoints(box)
        xmin=np.min(points[:,0]);xmax=np.max(points[:,0])
        ymin=np.min(points[:,1]);ymax=np.max(points[:,1])
        rect=[xmin,xmax,ymin,ymax]
        return rect

    def rpoints(self,box):
        cx,cy,w,h,angle=box
        angle=angle*math.pi/180
        p0x,p0y = self.rotatePoint(cx,cy, cx - w/2, cy - h/2, -angle)
        p1x,p1y = self.rotatePoint(cx,cy, cx + w/2, cy - h/2, -angle)
        p2x,p2y = self.rotatePoint(cx,cy, cx + w/2, cy + h/2, -angle)
        p3x,p3y = self.rotatePoint(cx,cy, cx - w/2, cy + h/2, -angle)
        points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
        points=np.array(points,dtype=int)
        return points

    def rotatePoint(self,xc,yc, xp,yp, theta):
        xoff = xp-xc
        yoff = yp-yc
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * xoff + sinTheta * yoff
        pResy = - sinTheta * xoff + cosTheta * yoff
        return xc+pResx,yc+pResy
        
    def restrict(self, rect,h,w):
        [xmin,xmax,ymin,ymax]=rect
        xmin=max(xmin,0);ymin=max(ymin,0)
        xmax=min(w-1,xmax);ymax=min(h-1,ymax)
        rect=[xmin,xmax,ymin,ymax]
        rect=list(map(int,rect))
        return rect
        
    def filter(self,box):
        _,_,w,h,_=box
        if max(w,h)>40:
            pass_filter=1
        else:
            pass_filter=0
        return pass_filter


def minirect(box, h, w):
    points = rpoints(box)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    rect = [xmin, xmax, ymin, ymax]
    rect = restrict(rect, h, w)
    return rect

def rpoints(box):
    cx, cy, w, h, angle = box
    angle = angle * np.pi / 180
    p0x, p0y = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
    p1x, p1y = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
    p2x, p2y = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
    p3x, p3y = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
    points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
    points = np.array(points, dtype=int)
    return points

def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return xc + pResx, yc + pResy

def restrict(rect, h, w):
    [xmin, xmax, ymin, ymax] = rect
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(w - 1, xmax)
    ymax = min(h - 1, ymax)
    rect = [xmin, xmax, ymin, ymax]
    rect = list(map(int, rect))
    return rect
