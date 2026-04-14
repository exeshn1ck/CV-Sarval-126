# -*- coding: utf-8 -*-
import numpy as np
import cv2
from Board import Board

class CircleBoard(Board):
    def __init__(self,
            pattern_size = (4, 11),
            circle_diameter = 0.015,
            circle_spacing = 0.02,
            save_images_path = None
        ):
        Board.__init__(self, pattern_size, save_images_path)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.circle_diameter = circle_diameter # in metres
        self.circle_spacing = circle_spacing # in metres

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 10000
        params.minThreshold = 1
        params.maxThreshold = 200
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.filterByInertia = True
        params.minInertiaRatio = 0.001
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.objp = self.create_board()

    def create_board(self):
        objp = np.zeros((self.pattern_size[1]*self.pattern_size[0],3), np.float32)
        temp_x = 0
        ind = 0
        for i in range(self.pattern_size[1]):
            start_y = 0
            if i%2 != 0:
                start_y += self.circle_spacing/2
            for k in range(self.pattern_size[0]):
                x = temp_x
                y = start_y + self.circle_spacing * k
                z = 0
                objp[ind] = (x,y,z)
                ind += 1
            temp_x = temp_x + (self.circle_spacing/2)
        return objp

    def get_board(self):
        return {
            'type': 'circle',
            'pattern_size': self.pattern_size,
            'circle_diameter': self.circle_diameter,
            'circle_spacing': self.circle_spacing,
        }

    def _find_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape

        ret, corners = cv2.findCirclesGrid(gray, self.pattern_size, blobDetector = self.detector, flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
        cv2.drawChessboardCorners(image, self.pattern_size, corners, ret)
        if ret == True and len(corners):
            return ret, corners, image_size
        return False, None, image_size
