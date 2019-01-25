#!/usr/bin/env pythonw

import os
import pickle
import cv2
import glob
import numpy as np


class CameraCalibrator:

    def __init__(self, input_images, nx, ny, debug=False):
        self.nx = nx  # the number of inside corners in x
        self.ny = ny  # the number of inside corners in y
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.input_images = input_images
        self.debug = debug
        self._get_obj_img_points()

    def save_calibration_pickle(self, img_path, output_pickle=''):
        img = cv2.imread(img_path)
        undist = self._cal_undistort(img, output_pickle)
        return undist

    def save_calibrated_images(self, output_dir):
        for img_path in self.input_images:
            output_path = os.path.join(output_dir, 'undistort_' + os.path.basename(img_path))
            img = cv2.imread(img_path)
            undist = self.save_calibration_pickle(img_path)
            combined = np.concatenate((img, undist), axis=1)
            cv2.imwrite(output_path, combined)

    def _get_obj_img_points(self):
        # Step through the list and search for chessboard corners
        for fname in self.input_images:
            self._draw_chessboard_corners(fname)
        if self.debug:
            cv2.destroyAllWindows()

    def _draw_chessboard_corners(self, fname):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            self.objpoints.append(objp)
            self.imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
            if self.debug:
                cv2.imshow('img', img)
                cv2.waitKey(500)

    def _cal_undistort(self, img, output_calibration_pickle=''):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[:-1], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        if len(output_calibration_pickle) > 0:
            dist_pickle = {}
            dist_pickle['mtx'] = mtx
            dist_pickle['dist'] = dist
            pickle.dump(dist_pickle, open(output_calibration_pickle, 'wb'))
        if self.debug:
            print(img.shape[1::-1])
            cv2.imshow('img', undist)
            cv2.waitKey(5000)
        return undist


if __name__ == '__main__':
    images = glob.glob('./camera_cal/calibration*.jpg')
    cal = CameraCalibrator(images, 9, 6)
    # save the calibration pickle file
    cal.save_calibration_pickle('./camera_cal/calibration1.jpg', './calibration_pickle.p')
    cal.save_calibrated_images('./output_images')
