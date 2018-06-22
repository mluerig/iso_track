# iso-track

## introduction

**iso-track** (isopod-tracking) is a semi-automated script that requires some user interaction to quantify movement of isopods (or other organisms) from video files. idea was based on https://github.com/approbatory/motion-tracker and implemented with https://github.com/soft-matter/trackpy. arena selector is based on Dan Masek's answer to https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python

**Please feel free to get in touch with me if you need help running the script or have questions about customizing it for your own study-system/organism: [contact @ eawag](http://www.eawag.ch/en/aboutus/portrait/organisation/staff/profile/moritz-luerig/show/)**

---

## installation

**required software:**

- python (3.6)
- opencv (3.3.1) + dependencies
- trackpy

install, for example, with anaconda (https://www.anaconda.com/download/):

```
conda install opencv numpy pandas trackpy
```

The script is not standalone, so a python interpreter needs to be used to modify and execute the script. Directories and input data need to be specified beforehand inside the script. At some point in the future I may provide a standalone executable version of the program.


---

## running the script...
... one code cell/lense at a time (cells are denoted by "#%%" and create a horizontal line in most IDEs)

1. load iso-track.py script
2. configure your directories and video settings (you can come back later if you need to change e.g. the kernel size or thresholding value)
3. import video_utils script (just a collection of custom functions we need to draw the arena)
4. draw arena. running this section will open a window, where you can select the arena to be included by left clicking. right click will complete the polygon and show you the result (green is included, red excluded in the motion analysis). 
5. open the video file. reads frame by frame (or every nth frame, if you chose to skip frames at 2.). shows you the live process (everything detected as moving gets white overlay) and saves the detected movements to a pandas dataframe. video of overlays is saved as well
6. calculates the trajectories. here you can play around a lot - see [trackpy reference ](http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.link_df.html). e.g. the larger the search range or history is, the more challenging it is for the algorithm to find a solution, especially if you have many moving objects in your video. if you have only one, it should be ok to go to high values
