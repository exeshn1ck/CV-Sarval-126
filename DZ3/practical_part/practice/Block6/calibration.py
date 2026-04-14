# -*- coding: utf-8 -*-
import argparse
import numpy as np
import json
import os.path
from datetime import datetime

from ChessBoard import ChessBoard
from CircleBoard import CircleBoard
from BaseMatrix import Base


class Calibration():
    def __init__(self, calib_input, source_type, board_type, save_images, all_frames):
        # get type of source
        self.calib_input = calib_input
        self.input_type = self.get_input_type(calib_input)
        self.all_frames = all_frames
        self.source_type = source_type

        if self.input_type == 'video':
            self.calib_input = int(self.calib_input) if len(self.calib_input)==1 else self.calib_input

        board_type = board_type.lower()
        if "chess" in board_type:
            self.board = ChessBoard(save_images_path=save_images)
        elif 'circle' in board_type:
            self.board = CircleBoard(save_images_path=save_images)
        else:
            self.board = Base()

    def get_input_type(self, input):
        if type(input) == str and os.path.isdir(input):
            return 'images'
        return 'video'

    def calibrate(self):
        if self.input_type == 'video':
            camera_calibration = self.board.calibrate_by_video(self.calib_input, self.source_type, self.all_frames)
        elif self.input_type == 'images':
            camera_calibration = self.board.calibrate_by_images(self.calib_input, self.source_type)
        else:
            print("Unable to calibrate, source not identified.")
        if camera_calibration is not None:
            calibration_data = {
                "cameras": [camera_calibration],
                "board": self.board.get_board()
            }
            return calibration_data


    def save(self, path, data):
        if data is None:
            print("Calibration data is None")
            return

        if not os.path.exists(path):
            os.makedirs(path)
            print("Create directory: " + path)

        now = datetime.now()
        now = now.strftime("%d-%m-%Y_%H-%M-%S")
        name = "calibration__%s.json"%str(now)
        full_path = os.path.join(path, name)
        with open(full_path, "w") as json_file:
            json.dump(data, json_file)

        print("Calibration data saved in: " + full_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--source', "-s", help="Cameras name or path to video or calibration images")
    parser.add_argument('--source_type', "-t", default='rgb', help="Cameras type")
    parser.add_argument('--board_type', "-b", default='chess', help="Chessboard or Circle board calibration")
    parser.add_argument('--save_path', default="./", help="Save path for calibration data")
    parser.add_argument('--save_images', default=None, help="Save path for calibration images")
    parser.add_argument('--all_frames', '-a', action="store_true", help="Calibration over all video frames")
    args = parser.parse_args()

    calibration = Calibration(args.source, args.source_type, args.board_type, args.save_images, args.all_frames)
    calibration_data = calibration.calibrate()
    calibration.save(args.save_path, calibration_data)
