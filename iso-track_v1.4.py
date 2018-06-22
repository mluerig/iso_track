#%%
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import pandas as pd
import trackpy as tp
import cv2
import os


#%% settings

os.chdir("E:/python1/iso-track")
os.listdir(os.getcwd())

video = "tadpole1.avi"

start = 0 # start video after x minutes
skip = 0 # number of frames to skip
roi = True # make video only of region of interest / selected polygon

# detection settings
kernelsize = 5 # for blurring
threshold = 200 # thresholding after blurring
#%% import and define custom functions

import video_utils
from video_utils import PolygonDrawer

def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]
def blur(image):
    return cv2.filter2D(image,ddepth,kern)

kernlen = kernelsize
kern = np.ones((kernlen,kernlen))/(kernlen**2)
ddepth = -1

#%% access video and draw arena

# draw arena
cap = cv2.VideoCapture(video)
idx = 0
while(cap.isOpened()):
    idx = idx + 1
    # extract frame from video stream
    ret, frame = cap.read()
    if idx == 10:
        break
poly = PolygonDrawer(video)
poly.run(frame)
arena_mask = cv2.cvtColor(poly.mask, cv2.COLOR_BGR2GRAY)
height, width, layers = frame.shape
cap.release()


rx,ry,rwidth,rheight = cv2.boundingRect(poly.points)


#%% background subtraction

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
if roi == True:
    video_out = cv2.VideoWriter("video_out.avi", fourcc, 25, (rwidth, rheight))
else:
    video_out = cv2.VideoWriter("video_out.avi", fourcc, 25, (width, height), False)

cap = cv2.VideoCapture(video)
fgbg = cv2.createBackgroundSubtractorMOG2()
idx1 = 0
idx2 = 0
df = pd.DataFrame()
ft = pd.DataFrame()
cap.get(cv2.CAP_PROP_FRAME_COUNT) 

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

    if idx1 > start * 1800: #and idx2 == 0:
        
        # read arena
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arena = np.bitwise_and(gray, arena_mask)
               
        # bg-fg subtractor
        fgmask = fgbg.apply(arena)
        
        # find contours
        mask = blur(fgmask)
        morph1 = cv2.erode(mask,np.ones((15,15),np.uint8),iterations = 1)
        morph2 = cv2.dilate(morph1,np.ones((5,5),np.uint8),iterations = 1)
        ret2, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cons = contours[1]
            
        # contour centerpoints
        scatter_l = list(map(avgit, cons))
        center=np.array(scatter_l)
    
    # use difference between two frames???
    
        if idx1 > start * 1800 and center.shape[0]>0:
            f = pd.DataFrame(center.reshape(center.shape[0],center.shape[2]), columns = list("xy"))
            if skip > 0:
                f["frame"] = idx1/skip         
            else:
                f["frame"] = idx1        
#            ft = ft.append(tp.locate(mask, 11, invert=True))
        df=df.append(f)
  
        img = cv2.addWeighted(arena, 1, mask, 0.5, 0)
        
        if roi == True:
            frame_out = img[ry:ry+rwidth,rx:rx+rheight]      
        else:
            frame_out = img

        cv2.namedWindow('overlay' ,cv2.WINDOW_NORMAL)
        cv2.imshow('overlay', frame_out)

        video_out.write(frame_out)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
video_out.release()
cv2.destroyAllWindows()

#%% movement analysis and plotting

traj = tp.link_df(df, 300, memory=300) #, neighbor_strategy="KDTree", link_strategy="recursive"

traj.to_csv("tadpole.csv", sep='\t')

plot = tp.plot_traj(traj, superimpose=arena)
fig = plot.get_figure()
fig.savefig("tadpole.png")

