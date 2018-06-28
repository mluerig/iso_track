# -*- coding: utf-8 -*-
"""
Created: 2018/06/22
Last Update: 2018/06/28
Version 0.1.4
@author: Moritz Lürig
"""

#%%
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import pandas as pd
import trackpy as tp
import cv2
import os


#%% settings

main_dir = "E:\\python1\iso-track\\sandbox"
video_in = "F:\\4_FID52nurAssel.MP4"

os.chdir(main_dir)

video_name = os.path.splitext(os.path.basename(video_in))[0]
video_out = video_name + "_out.avi"

start = 0.5 # start frame capture after x minutes | use to cut out handling time in the beginning
skip = 10 # number of frames to skip (1 = every second frame, 2 = every third frame, ...) | useful when organisms are moving too slow
roi = True # make video only of region of interest / selected polygon

# detection settings

#blurring
blur_kern = 10 # for blurring | higher = more coarsely grained 
blur_thresh = 90 # thresholding after blurring | higher = smaller area
kern = np.ones((blur_kern,blur_kern))/(blur_kern**2)
ddepth = -1

dilate_kern = 7
dilate_iter = 2
min_area=50

#%% import and define custom functions

import video_utils
from video_utils import PolygonDrawer

def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]
def blur(image):
    return cv2.filter2D(image,ddepth,kern)

#%% access video and draw arena

# draw arena
cap = cv2.VideoCapture(video_in)
idx = 0
while(cap.isOpened()):
    idx = idx + 1
    # extract frame from video stream
    ret, frame = cap.read()
    if idx == 10:
        break
poly = PolygonDrawer(video_in)
poly.run(frame)
arena_mask = cv2.cvtColor(poly.mask, cv2.COLOR_BGR2GRAY)
height, width, layers = frame.shape
cap.release()

rx,ry,rwidth,rheight = cv2.boundingRect(poly.points)
cv2.imwrite(os.path.join(main_dir, video_name + "_arena.png"), poly.arena)    


#%% background subtraction

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc('F','M','P','4')

if roi == True:
    video_out_writer = cv2.VideoWriter(os.path.join(main_dir,  video_out), fourcc, 25, (rwidth, rheight), False)
else:
    video_out_writer = cv2.VideoWriter(os.path.join(main_dir,  video_out), fourcc, 25, (width, height), False)

cap = cv2.VideoCapture(video_in)
fgbg = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold = 12, detectShadows = True)
idx1 = 0
idx2 = 0
df = pd.DataFrame()
ft = pd.DataFrame()
nframes=cap.get(cv2.CAP_PROP_FRAME_COUNT) 

while(cap.isOpened()):

    
    # start video capture (after x minutes)
    ret, frame = cap.read()
    if ret==False:
        break

    idx1 = idx1 + 1  
    idx2 = idx2 + 1
    if idx2 >= skip:
        idx2 = 0
    print(idx1)

    if idx1 > start * 1800 and idx2 == 0:
        
        # read arena
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arena = np.bitwise_and(gray, arena_mask)
               
        # bg-fg subtractor
        fgmask = fgbg.apply(arena)
        
        # find contours
        mask = blur(fgmask)
        ret, thresh = cv2.threshold(mask, blur_thresh, 255, cv2.THRESH_BINARY)
        morph = cv2.dilate(thresh,np.ones((dilate_kern,dilate_kern),np.uint8),iterations = dilate_iter)
        contours = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours = contours[1]
        
 #       cv2.imshow('overlay', thresh)

        
        # filter contours
        contours_good=[]
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > min_area:
                contours_good.append(cont)

        # contour centerpoints
        scatter_l = list(map(avgit, contours_good))
        center=np.array(scatter_l)
    
        # collect center points of detected contours in data frame    
        if idx1 > start * 1800 and center.shape[0]>0:
            f = pd.DataFrame(center.reshape(center.shape[0],center.shape[2]), columns = list("xy"))
            if skip > 0:
                f["frame"] = idx1/skip         
            else:
                f["frame"] = idx1        
#            ft = ft.append(tp.locate(mask, 11, invert=True))
        df=df.append(f)
  
        # show only selected arena
        img = cv2.addWeighted(arena, 1, thresh, 0.5, 0)
        if roi == True:
            frame_out = img[ry:ry+rheight,rx:rx+rwidth]      
        else:
            frame_out = img

        # show and write new frames
        cv2.namedWindow('overlay' ,cv2.WINDOW_NORMAL)
        cv2.imshow('overlay', frame_out)
        video_out_writer.write(frame_out)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
video_out_writer.release()
cv2.destroyAllWindows()

#%% movement analysis and plotting

# find trajectories - check http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.link_df.html for all options
traj = tp.link_df(df, search_range = 75, memory=250, neighbor_strategy="KDTree", link_strategy="nonrecursive")
traj_filter = tp.filtering.filter_stubs(traj, threshold=100)

# plot trajectories
plot = tp.plot_traj(traj, superimpose=arena)
fig = plot.get_figure()
fig.savefig(os.path.join(main_dir, video_name + "trajectories.png"), dpi=500)

# save trajectories to csv

traj.to_csv(os.path.join(main_dir, video_name + "_trajectories.csv"), sep='\t')