# iso_track

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [introduction](#introduction)
- [installation](#installation)
- [running the script...](#running-the-script)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

---

## introduction

**iso_track** (**iso**pod_**track**ing) is a semi-automated script to quantify movement of animals in videos using foreground-background detection in response to pixel-motion ([opencv fgbg-subtractor](https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html)). It has been sucessfully used to quantify movement of the freshwater isopod *Asellus aquaticus* (for the scientific background refer to http://luerig.net/Research/#Isopods) and threespine stickleback (*Gasterosteus aculeatus*). The idea is based on [approbatory/motion-tracker](https://github.com/approbatory/motion-tracker) and was implemented with [soft-matter/trackpy](https://github.com/soft-matter/trackpy). Arena selector is based on Dan Masek's answer to [this SO question](https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python).

| [<img src="assets/iso_track_demo.gif" width="100%" />](https://vimeo.com/283075068) | 
|:--:| 
|**Example of iso_track motion detection in a laboratory foraging experiment**. Note that the drawn overlays are not the actual analysis, but serve the purpose of evaluating the overall detection quality. The trackpy routine eliminates all glitches (flickering isopod or fish overlays) |

The script uses different *modules* to quantify movement of different animals at the same time. In the given example, the goal is to quantify isopod movement in response to movements of a predatory fish (stickleback). However, the two movement patterns are very different and would be difficult to detect from a single routine. So each animal has its own code-module that contains the necessary adjustments of the detected foreground. 

| <img src="assets/trajectories.png" width="50%"/><img src="assets/movement.png" width="50%" /> | 
|:--:| 
|**Example of iso_track output. Left:** fish movement over 15 minutes (from green to blue: earlier to later in time). Most glitches have been removed to this step. The trajectories can be smoothed (later in R or Python), e.g. by averaging over multiple frames per second. **Right:**  Summary of fish and isopod movement. This figure shows how much all isopods have moved (green line: the sum of the movement of all isopods in one second) and how much the fish has moved (red line: mean fish movement in one second).|

**It's easy to add a module that can quantifiy movement of other animals in different environments. Please feel free to get in touch with me if you need help running the script or have questions about customizing it for your own study-system: [moritz lürig @ eawag](http://www.eawag.ch/en/aboutus/portrait/organisation/staff/profile/moritz-luerig/show/)**

---

## installation

**required software:**

- python (3.6)
- opencv (3.3.1) + dependencies (more info here: https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda)
- trackpy (more info here http://soft-matter.github.io/trackpy/v0.3.0/installation.html)

install, for example, with anaconda (https://www.anaconda.com/download/):

```
conda update conda
conda install numpy pandas 
conda install -c conda-forge opencv 
conda install -c soft-matter trackpy
```

**IMPORTANT: The script is not executable standalone, so a python interpreter has to be used to modify and execute the script (e.g. [Spyder](https://github.com/spyder-ide/spyder)). Directories and input data need to be specified beforehand inside the script. At some point in the future I may provide a standalone executable version of the program.**

---

## running the script... 
... one code cell/lense at a time (cells are denoted by "#%%" and create a horizontal line in most IDEs)

(refer to the annotation inside the script for more details)

1. download [iso_track.py](iso_track.py) and [iso_track_modules.py](iso_track_moduls.py) scripts into your current working directory 
2. open iso_track.py, configure your current wd with `os. chdir()` and import the modules
3. configure video and detection settings. `backgr_thresh` will have the greatest effect on your results, as it defines the sensitivity of the foreground-background detector. lower values increase sensitivity (i.e. more likely to detect something), but also increase noise. `skip` frames if your organisms are moving too slow. configure the modules separately: the blurring kernels (`blur_kern`) and iterations  (`blur_iter`) will smoothe the detected contours. `min_length` and `max_length` can be to exclude more noise. `shadow` detection improves results but slowes the detection. you can (and should) come back here often to improve your results
4. draw arena (will remove a lot of nois). running this section will open a window, where you can select the arena to be included by left clicking. right click will complete the polygon and show you the result (green is included, red excluded in the motion analysis). 
5. everything is set up now - run the video capture! reads frame by frame (or every nth frame, if you chose to skip frames at 2.). shows you the live process (everything detected as moving gets white overlay) and saves the detected movements to a pandas dataframe. video of overlays is saved as well
6. calculates the trajectories. here you need to find out what works best for your case - see [trackpy reference ](http://soft-matter.github.io/trackpy/v0.3.0/generated/trackpy.link_df.html). e.g. the larger the `search_range` or `memory` is, the more challenging it is for the algorithm to find a solution, especially if you have many moving objects in your video. if you have only one, it should be ok to go to high values. the filtering step is optional, but can be useful to eliminate spurious trajectories. saves pictures and tables of trajectories to your main working dir.
