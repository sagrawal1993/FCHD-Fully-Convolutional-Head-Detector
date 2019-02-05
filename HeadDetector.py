from __future__ import division

import os
import torch as t
from src.config import opt
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt 
import src.array_tool as at
#from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from src.config import opt
import cv2
import time

SAVE_FLAG = 1
THRESH = 0.01
IM_RESIZE = False

class HeadDetection:
    def __init__(self, model_path="/home/ubuntu/suraj/package/FCHD-Fully-Convolutional-Head-Detector/checkpoints/head_detector_final"):
        self.head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
        trainer = Head_Detector_Trainer(self.head_detector).cuda()
        trainer.load(model_path)

    def find_head(self, cv2_im):
        """
        This function will filter background of person's body.
        """
        f = Image.fromarray(cv2_im)
        f.convert('RGB')
        img_raw = np.asarray(f, dtype=np.uint8)
        img_raw_final = img_raw.copy()
        img = np.asarray(f, dtype=np.float32)
        img = img.transpose((2,0,1))
        _, H, W = img.shape
        img = preprocess(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        img = at.totensor(img)
        img = img[None, : ,: ,:]
        img = img.cuda().float()
        st = time.time()
        pred_bboxes_, _ = self.head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)
        et = time.time()
        tt = et - st
        print ("[INFO] Head detection over. Time taken: {:.4f} s".format(tt))
        box_list = []
        for i in range(pred_bboxes_.shape[0]):
            (y1, x1, y2, x2) = pred_bboxes_[i,:]/scale
            box_list.append((x1, y1, x2-x1, y2-y1))
        return box_list

def processsVideo(video_path, output_file_name):
    detector = HeadDetection()
    is_a_vertical_vid = False
    vid_t = cv2.VideoCapture(video_path)
    fps = vid_t.get(cv2.CAP_PROP_FPS)
    frame_width = int(vid_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # use X264  for lower size
    fourcc = cv2.VideoWriter_fourcc(*'xvid')
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (frame_width, frame_height))
    total_frames = 0
    while vid_t.isOpened():
        ret, frame = vid_t.read()
        if ret is False:
            break
        if is_a_vertical_vid:
            # vid.append(encode_np_image(np.rot90(frame,1,(1,0)).copy()))
            curr_frame = np.rot90(frame, 1, (1, 0)).copy()
        else:
            # vid.append(encode_np_image(frame))
            curr_frame = frame.copy()
        img = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        box_list = detector.find_head(img)
        for box in box_list:
            cv2.rectangle(img, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), (0, 0, 255), 1)
        out.write(img)
        total_frames += 1
    out.release()

def processImage(img_path):
    detector = HeadDetection()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box_list = detector.find_head(img)
    for box in box_list:
        cv2.rectangle(img, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), (255, 0, 0), 1)
    cv2.imwrite("output.jpg", img)

processsVideo("/home/ubuntu/suraj/data/countingData/ShopEntrance.mp4","video.mp4")
#processImage("img.jpg")
