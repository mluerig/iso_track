# -*- coding: utf-8 -*-
"""
Created: 2018/06/22
Last Update: 2018/06/29
Version 0.1.5
@author: Moritz Lürig
"""

#%%
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import pandas as pd
import trackpy as tp
import cv2
import os


#%% directories

os.chdir("E:\\python1\\iso-track")
main_dir = "E:\\python1\\iso-track\\example"
video_in = "E:\\python1\\iso-track\\example\\asellus-sample-1.mp4"

#%% settings

video_name = os.path.splitext(os.path.basename(video_in))[0]
video_out = video_name + "_out.avi"

# video settings

start = 0 # MINUTES wait before capture after is started | use to cut out handling time in the beginning
skip = 2 # nFRAMES to skip (1 = capture every second frame, 2 = every third frame, ...) | useful when organisms are moving too slow
roi = False # make video only of region of interest / selected polygon
wait = 7.5 # | MINUTES how long can organism sit still

# detection settings

# background-subtractor
backgr_thresh = 12 # lower = more of organism is detected (+ more noise)
shadows = True # detect shadows
min_area=50 # | minimum px area to be included 


#blurring
blur_kern = 10 # for blurring | higher = more coarsely grained 
blur_thresh = 90 # thresholding after blurring | higher = smaller area
kern = np.ones((blur_kern,blur_kern))/(blur_kern**2)
ddepth = -1

#add borders to mask
dilate_kern = 7
dilate_iter = 2

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

nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
fps = cap.get(cv2.CAP_PROP_FPS)
vid_length = str(int(( nframes / fps)/60)).zfill(2) + ":" +str(int((((nframes / fps)/60)-int((nframes / fps)/60))*60)).zfill(2)

history = (wait*60) * (fps / skip)
fgbg = cv2.createBackgroundSubtractorMOG2(history = int(history), varThreshold = int(backgr_thresh), detectShadows = shadows)
idx1 = 0
idx2 = 0
df = pd.DataFrame()
ft = pd.DataFrame()

while(cap.isOpened()):

    
    # start video capture (after x minutes)
    ret, frame = cap.read()
    capture = False
    
    if ret==False:
        break

    idx1 = idx1 + 1  
    idx2 = idx2 + 1
    if idx2 >= skip:
        idx2 = 0
        
    mins = str(int((idx1 / fps)/60)).zfill(2)
    secs = str(int((((idx1 / fps)/60)-int(mins))*60)).zfill(2)
    
    if idx1 > start * 1800 and idx2 == 0:
        
        capture = True
        
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
        
    if capture == True:
        print(mins + ":" + secs + " / " + vid_length + " - " + str(idx1) + " / " + str(int(nframes)) + " - captured" )
    else:
        print(mins + ":" + secs + " / " + vid_length + " - " + str(idx1) + " / " + str(int(nframes)))

cap.release()
video_out_writer.release()
cv2.destroyAllWindows()

#%% movement analysis and plotting

# find trajectories - check http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.link_df.html for all options
traj = tp.link_df(df, search_range = 50, memory=300, neighbor_strategy="KDTree", link_strategy="nonrecursive")
traj_filter = tp.filtering.filter_stubs(traj, threshold=20)

# plot trajectories
plot = tp.plot_traj(traj, superimpose=arena)
fig = plot.get_figure()
fig.savefig(os.path.join(main_dir, video_name + "_trajectories.png"), dpi=300)

# save trajectories to csv

traj.to_csv(os.path.join(main_dir, video_name + "_trajectories.csv"), sep='\t')