# -*- coding: utf-8 -*-
"""
Created: 2018/06/22
Last Update: 2018/08/03
Version 0.2
@author: Moritz Lürig
"""

#%% 
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import cv2
import os

#%% directories

os.chdir("E:\\python1\\iso_track")
main_dir = "E:\\python1\\iso_track\\example"
video_path = "E:\\python1\\iso_track\\example\\iso_track_example_vid.mp4"

video_name = os.path.splitext(os.path.basename(video_path))[0]

from iso_track_modules import polygon_drawer, video_info, video_time, capture_feedback # support modules
from iso_track_modules import fish_module, isopod_module # analysis modules

#%% settings

# video settings 
start = 0 # MINUTES wait before capture after is started | use to cut out handling time in the beginning
skip = 4 # nFRAMES to skip (1 = capture every second frame, 2 = every third frame, ...) | useful when organisms are moving too slow
roi = False # make video only of region of interest / selected polygon
wait = 2.5 # | MINUTES how long can organism sit still
scale = 4.315 # px/mm

# background-subtractor
backgr_thresh = 10 # lower = more of organism is detected (+ more noise)

# fish module
blur_kern_fish = 25 # for blurring | higher = more coarsely grained 
blur_thresh_fish = 60 # thresholding after blurring | higher = smaller area
min_length_fish = 100 # | minimum ellupse length to be included 
shadows_fish = True

# isopod module
blur_kern_iso = 25 # for blurring | higher = more coarsely grained 
blur_thresh_iso = 40 # thresholding after blurring | higher = smaller area
max_length_iso = 80 # | minimum ellipse length to be included 
min_length_iso = 15
shadows_isopod = False

#%% extract video information, add arenas, configure video output

v = video_info(video_path)

arena = polygon_drawer(v.name, main_dir)      
arena.run(v.frame)

fourcc = cv2.VideoWriter_fourcc(*"DIVX")
video_out_writer = cv2.VideoWriter(os.path.join(main_dir,  v.name + "_out.avi"), fourcc, v.fps, (v.width, v.height), True)
 
fgbg = cv2.createBackgroundSubtractorMOG2(history = int((wait*60) * (v.fps / skip)), varThreshold = int(backgr_thresh), detectShadows = True)


#%%

# intiate
idx1, idx2 = (0,0)
df_fish, df_isopod  = ( pd.DataFrame(),pd.DataFrame())
cap = cv2.VideoCapture(video_path)

# start
while(cap.isOpened()):
    
    # read video 
    ret, frame = cap.read()     
    capture = False
    if ret==False:
        break
    
    # indexing 
    idx1, idx2 = (idx1 + 1,  idx2 + 1)    
    if idx2 == skip:
        idx2 = 0    
        
    # engange  fgbg-algorithm shortly before capturing 
    if idx1 > (start * v.fps * 60) - (3*v.fps):
        fgmask = fgbg.apply(frame)
        frame_out = frame
        
    # start modules after x minutes
    if idx1 > (start * v.fps * 60) and idx2 == 0:
        capture = True
               
        # =============================================================================
        # FISH MODULE
        # =============================================================================
        fish = fish_module(frame, fgmask, shadows_fish, blur_kern_fish, blur_thresh_fish, min_length_fish)   
        if not fish.empty :
            f = pd.DataFrame(data=fish.center, columns = list("xy"))
            if skip > 0:
                f["frame"] = idx1/skip         
            else:
                f["frame"] = idx1        
            df_fish=df_fish.append(f)      
        frame_out = fish.draw(frame_out, ["contour", "ellipse", "text"])

        
        # =============================================================================
        # ISOPOD MODULE
        # =============================================================================
        if not fish.empty:
            fgmask = np.bitwise_and(fgmask, fish.box) # exclude fish area
        
        isopod = isopod_module(frame, fgmask, shadows_isopod, blur_kern_iso, blur_thresh_iso, min_length_iso, max_length_iso, arena.mask_gray)  
        if not isopod.empty:
            f = pd.DataFrame(data=isopod.center, columns = list("xy"))
            if skip > 0:
                f["frame"] = idx1/skip         
            else:
                f["frame"] = idx1        
            df_isopod=df_isopod.append(f)      
        frame_out = isopod.draw(frame_out, ["contour", "ellipse", "text"]) #, 
        
        # show output image and write to file
        cv2.namedWindow('overlay' ,cv2.WINDOW_NORMAL)
        cv2.imshow('overlay',  frame_out)    
        video_out_writer.write(frame_out)
        
        # keep stream open
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # return present time and captured frames 
    t = video_time(idx1, v.fps)
    capture_feedback(t.mins, t.secs, idx1, v.length, v.nframes, capture)
    
    
cap.release()
video_out_writer.release()
cv2.destroyAllWindows()


#%% trackpy-routine - trajectories, pictures, dataframe and saving
            
# find trajectories - check http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.link_df.html for all options
traj_fish = tp.link_df(df_fish, search_range = 200, memory=50, neighbor_strategy="KDTree", link_strategy="nonrecursive")
traj_fish_filter = tp.filtering.filter_stubs(traj_fish, threshold=20)

df_isopod_counts = df_isopod["frame"].value_counts()
df_isopod_filtered = df_isopod.groupby("frame").filter(lambda x: len(x) < 22)  
traj_isopod = tp.link_df(df_isopod_filtered, search_range = 50, memory=500, neighbor_strategy="KDTree", link_strategy="nonrecursive")
traj_isopod_filter = tp.filtering.filter_stubs(traj_isopod, threshold=20)

# plot fish trajectories
plot = tp.plot_traj(traj_fish_filter, superimpose=v.frame)
fig1 = plot.get_figure()
fig1.savefig(os.path.join(main_dir, video_name + "_fish_trajectories.png"), dpi=300)
plt.close('all')

# plot isopod trajectories
plot = tp.plot_traj(traj_isopod_filter, superimpose=v.frame, colorby="particle")
fig2 = plot.get_figure()
fig2.savefig(os.path.join(main_dir, video_name + "_isopod_trajectories.png"), dpi=300)
plt.close('all')

# save isopod trajectories to csv
traj_fish.to_csv(os.path.join(main_dir, video_name + "_fish_trajectories_full.csv"), sep='\t')
traj_isopod.to_csv(os.path.join(main_dir, video_name + "_isopod_trajectories_full.csv"), sep='\t')


