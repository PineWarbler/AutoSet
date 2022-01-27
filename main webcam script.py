# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:01:09 2021

@author: Peter Reynolds
"""

print('importing modules...')
from Set_Helper_Functions import set_res, analyzePhoto, printRuntime, undistort
import numpy as np
import cv2
import time
start = time.time() 


webcamXResolution = 1920
webcamYResolution = 1080

DIM=(webcamXResolution, webcamYResolution)

# fisheye distortion correction array unique to the camera used.  In my case...
K = np.array([[1379.272132630417, 0.0, 1028.5163535739657], [0.0, 1378.3566387211583, 525.4940091348946], [0.0, 0.0, 1.0]])
D = np.array([[-0.09866785524092823], [0.5186625294188112], [-2.695194292494894], [3.725662209571771]])

print('setting up camera...')
cap = cv2.VideoCapture(0)

# enable full resolution of webcam
set_res(cap, webcamXResolution, webcamYResolution)


if __name__ == "__main__":
    print('commencing image looping...')
    while True:  
        trigger = cv2.waitKey(1)
            
        ret, frame = cap.read()

        cv2.imshow('frame', undistort(frame)) # if showing de-fisheyed frame

            
        if trigger == ord('q'): # the keyboard key 'q' is the trigger to terminate video reading loop
            cap.release()
            cv2.destroyAllWindows()
            print('Loop terminated.')
            break
        
        if trigger == ord('p'):  # the keyboard key 'p' is the trigger to take a picture from video stream
            ret, frame = cap.read() # store the current frame of the video stream

            pictureFrame = undistort(frame) # de-fisheye the frame

            cv2.destroyAllWindows()

            analyzePhoto(pictureFrame) # send frame to `analyzePhoto` function to be analyzed for sets

    printRuntime(start, time.time())