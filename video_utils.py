# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:26:13 2018
based on https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python
@author: Dan Mašek, Moritz Lürig
"""

import numpy as np
import cv2
import copy

FINAL_LINE_COLOR = (0, 255, 0)
FILL_COLOR = (255,255,255)
WORKING_LINE_COLOR = (255, 0, 0)


class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon

    # mouse action
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    # draw lines
    def run(self, image):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        temp_canvas = copy.deepcopy(image)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(temp_canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(temp_canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, temp_canvas)
            temp_canvas = copy.deepcopy(image)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # create final arena
        canvas = image
        
        # of a filled polygon
        zeros = np.zeros(canvas.shape, np.uint8)
        red = np.zeros(canvas.shape, np.uint8)
        red[:,:,2] = 255
        
        if (len(self.points) > 0):
            cv2.fillPoly(red, np.array([self.points]), FINAL_LINE_COLOR)
               
        if (len(self.points) > 0):
            cv2.fillPoly(zeros, np.array([self.points]), FILL_COLOR)
            
        self.mask = zeros
        self.arena = cv2.addWeighted(copy.deepcopy(canvas), .8, red, 0.2, 0)

        # show arena
        cv2.imshow(self.window_name, self.arena)

        # Waiting for the user to press any key and return points
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        print("Polygon = %s" % self.points)

        self.points = np.array([self.points])


