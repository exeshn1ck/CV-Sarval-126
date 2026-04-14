# -*- coding: utf-8 -*-
import numpy as np
import cv2
from Board import Board

class ChessBoard(Board):
    def __init__(self,
            pattern_size = (9, 6),
            square_size = 0.025,
            save_images_path = None
        ):
        Board.__init__(self, pattern_size, save_images_path)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

        self.square_size = square_size # in metres

        self.objp = np.zeros((self.pattern_size[1]*self.pattern_size[0],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.pattern_size[0],0:self.pattern_size[1]].T.reshape(-1,2)*self.square_size

    def get_board(self):
        return {
            'type': 'chess',
            'pattern_size': self.pattern_size,
            'square_size': self.square_size,
        }

    def _find_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape

        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        cv2.drawChessboardCorners(image, self.pattern_size, corners, ret)

        if ret == True and len(corners):
            corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
            return ret, corners, image_size
        return False, None, image_size
