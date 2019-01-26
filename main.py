#!/usr/bin/env pythonw

import glob
import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

from lane_detector import LaneDetector


def get_output_path(input_path, prefix='_'):
    return './output_images/' + prefix + os.path.basename(input_path)


def save_image(path, imgs, prefix='_', time=10000):
    combined = np.concatenate(imgs, axis=1)
    cv2.imwrite(get_output_path(path, prefix), combined)
    cv2.waitKey(time)


def detect_lane_for_video(input_video):
    # Create LaneTracker object with matrix and distortion coefficients
    lane_detector = get_lane_detector()

    # Name of output video after applying lane lines
    output_video = 'laneDetect_' + os.path.basename(input_video)
    # Apply lane lines to each frame of input video and save as new video file
    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(lane_detector.apply_lines)
    video_clip.write_videofile(output_video, audio=False)


def get_lane_detector():
    pickle_file = './calibration_pickle.p'
    if not os.path.isfile(pickle_file):
        # get the calibration data calibration_pickle.p
        os.system('./camera_calibrator.py')

    dist_pickle = pickle.load(open('./calibration_pickle.p', 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    lane_detector = LaneDetector(mtx, dist)
    return lane_detector


if __name__ == '__main__':
    detect_lane_for_video('./project_video.mp4')
    #detect_lane_for_video('./challenge_video.mp4')
    #detect_lane_for_video('./harder_challenge_video.mp4')
    # detect_lane_for_video(sys.argv[1])
    if 0:
        lane_detector = get_lane_detector()
        for path in glob.glob('./test_images/*.jpg'):
            img = cv2.imread(path)
            result = lane_detector.apply_lines(img)
            cv2.imwrite(get_output_path(path, 'final_'), result)
