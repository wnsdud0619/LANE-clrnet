#!/usr/bin/env python
from ctypes import sizeof
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import numpy as np
from PIL import Image
import torch
import cv2
import os
import rospy
import os.path as osp
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
import math


#ROS Imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose,PoseArray,Point,Quaternion


#package = RosPack()
#package_path = package.get_path('Lane_detection')

class Detect(object):
    def __init__(self):
        cfg = Config.fromfile("/home/dgist/clrnet_ws/src/clrnet/src/configs/clrnet/clr_dla34_culane.py")
        cfg.show = True
        cfg.savedir = "/home/dgist/test"
        # cfg.load_from = "/home/ajay/clrnet/models/culane_r101.pth"  
        cfg.load_from = "/home/dgist/clrnet_ws/src/clrnet/src/CULane.pth"   

        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        self.bridge = CvBridge()
        load_network(self.net, self.cfg.load_from)
        self.image_sub = rospy.Subscriber("/sensing/camera/traffic_light/image_raw", Image, self.imageCb, queue_size = 10, buff_size = 2**24)
        self.image_pub = rospy.Publisher("/out_image", Image, queue_size=10)
        self.lanepoints_pub = rospy.Publisher("/lane_points", PoseArray, queue_size=10)
        # self.lanepoints_pub = rospy.Publisher("/lane_points", LaneDataArray, queue_size=10)
        
        # intrinsic = np.load("/home/ajay/lanedet/camera_data/cam_mtx.npy")
        # R_mtx=np.load("/home/ajay/lanedet/camera_data/R_mtx.npy")
        # tvec1=np.load("/home/ajay/lanedet/camera_data/tvec1.npy")
        # print("Intrinsic",intrinsic)
        # print("R_mtx",R_mtx)
        # print("tvec1",tvec1)
        

    def preprocess(self, input_img):
        # ori_img = cv2.imread(img_path)
        ori_img = input_img
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        # data.update({'img_path':img_path, 'ori_img':ori_img})
        data.update({'input_img':input_img, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        return data


    def line_interpolate(self,lane):
        # print(" Before Interpolate Size of Lane: ", len(lane))
        for i in range(len(lane)-1):
            x0, y0 = lane[i]
            x1, y1 = lane[i+1]                
            limit = math.fabs(y1-y0)
            # print(i, limit)
            for j in range(int(limit)):
                y=y0+j
                x=x0+((x1-x0)/(y1-y0))*(y-y0)
                lane= np.append(lane, np.array([[x, y]]), axis=0)

        return lane      

    def show(self, data):
        # out_file = self.cfg.savedir 
        # if out_file:
        #     out_file = osp.join(out_file, osp.basename(data['input_img']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        lane_msg = PoseArray()
        lane_points =[]
        # print("Number of Lane: ", len(lanes))
        # print("------------------------------------------------------------------")
        for lane in lanes:            
            # print("lane size Before: ",len(lane))
            lane=self.line_interpolate(lane)
            # print("lane size After: ",len(lane))
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                lane_points.append(Pose(position=Point(x=x,y=y,z=0),orientation=Quaternion(x=0,y=0,z=0,w=1)))

        
        lane_msg.poses = lane_points
        # print("Lane size: ",len(lane_msg.poses))
        self.lanepoints_pub.publish(lane_msg)
        imshow_lanes(data['ori_img'], lanes)
        
    
    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data

    def imageCb(self, img_msg):
              
        ncols = img_msg.width
        nrows = img_msg.height
        np_arr = np.frombuffer(img_msg.data, np.uint8).reshape(nrows, ncols, 3)
        input_img=cv2.resize(np_arr, (800, 320), interpolation = cv2.INTER_AREA)
        detect.run(input_img)

        # cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        # cv2.imshow("output", input_img)
        # cv2.waitKey(3)
        return True

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process():
    detect = Detect()
    # paths = get_img_paths("/home/ajay/lanedet/data/dgist/")
    paths = get_img_paths("/home/ajay/clrnet/data/culane/")
    for p in tqdm(paths):
        detect.run(p)
        
if __name__ == '__main__':
    rospy.init_node("Lane_detection")
    detect = Detect()
    # process()
    rospy.spin()
