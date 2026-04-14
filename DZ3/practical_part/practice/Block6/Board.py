# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os.path
import glob
import uuid

class Board():
    def __init__(self,
            pattern_size,
            save_images_path = None
        ):
        self.pattern_size = pattern_size
        self.save_images_path = save_images_path

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.save_images_path = save_images_path
        if self.save_images_path is not None and not os.path.exists(self.save_images_path):
            os.makedirs(self.save_images_path)
            print("Create directory: " + self.save_images_path)

    def _find_board(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape
        return False, None, image_size

    def _calibrate(self, objpoints, imgpoints, image_size, calib_source, type):
        print("Calibration...")
        resolution = {'h': image_size[0], 'w': image_size[1]}
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                image_size[::-1],
                None,
                None,
                self.criteria,
                flags=(cv2.CALIB_FIX_PRINCIPAL_POINT)
            )

        if ret:
            print("-"*20)
            print("Camera Matrix: ", cameraMatrix)
            print("Distortion Coefficients : ", distCoeffs)
            total_error = self.total_error(objpoints, imgpoints, cameraMatrix, distCoeffs, rvecs, tvecs)
            print("-"*20)

            optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix,
                distCoeffs,
                (resolution['w'], resolution['h']),
                1,
                (resolution['w'], resolution['h']),
                )
            return {
                "id": str(uuid.uuid4()),
                "type": type,
                "calibration_source": calib_source,
                "camera_matrix": cameraMatrix.tolist(),
                "optimal_camera_matrix": optimal_camera_matrix.tolist(),
                "roi": roi,
                "distortion": distCoeffs.tolist(),
                "rvecs": [vec.tolist() for vec in rvecs],
                "tvecs": [vec.tolist() for vec in tvecs],
                "resolution": resolution,
                "total_error": total_error
            }
        print("Calibration failed")
        return None

    def calibrate_by_video(self, calib_source, type="rgb", all_frames = True):
        cap = cv2.VideoCapture(calib_source)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        frame_counter = 0
        while True:
            save_flag = True if all_frames else False
            flag, img = cap.read()
            if not flag:
                break
            clear_img = img.copy()

            ret, corners, image_size = self._find_board(img)

            # resize only for show image
            img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
            # show number frames for calibration
            self._draw_label(img, frame_counter)
            cv2.imshow('frame', img)
            ch = cv2.waitKey(1)
            if ch == 13:
                if not all_frames:
                    save_flag = True
                    if ret and corners is not None:
                        objpoints.append(self.objp)
                        imgpoints.append(corners)
                        frame_counter += 1

            if ch == ord('q') or ch == 27:
                break

            if all_frames and ret and corners is not None:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                frame_counter += 1

            # save frame
            if save_flag and self.save_images_path is not None:
                name = str(frame_counter) + ".jpeg"
                cv2.imwrite(os.path.join(self.save_images_path, name), clear_img)

        cv2.destroyAllWindows()
        cap.release()

        if len(objpoints) and len(imgpoints):
            return self._calibrate(objpoints, imgpoints, image_size, calib_source, type)
        print("Corners not find")
        return None

    def calibrate_by_images(self, calib_source, type = 'rgb'):
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(calib_source + "*.jpg") + glob.glob(calib_source + "*.jpeg")
        for fname in images:
            img = cv2.imread(fname)

            ret, corners, image_size = self._find_board(img)
            if ret and corners is not None:
                objpoints.append(self.objp)
                imgpoints.append(corners)

            # resize only for show image
            img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)
            cv2.imshow('img', img)
            ch = cv2.waitKey(1)
            if ch == ord('q') or ch == 27:
                break

        cv2.destroyAllWindows()

        if len(objpoints) and len(imgpoints):
            return self._calibrate(objpoints, imgpoints, image_size, calib_source, type)
        print("Corners not find")
        return None

    def total_error(self, objpoints, imgpoints, mtx, dist, rvecs, tvecs):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        total_error = mean_error/len(objpoints)
        print("Total error: ", total_error)
        return total_error

    def _combine_images(self, img1, img2, axis = 0, space = 0):
        if(space != 0):
            (height, width, d) = img1.shape
            if(axis):
                space_img = np.zeros((height,space,3), np.uint8)
            else:
                space_img = np.zeros((space,width,3), np.uint8)
            return np.concatenate((img1, space_img, img2), axis) 
        else:
            return np.concatenate((img1, img2), axis)

    def _draw_label(self, img, frame_counter):
        # show number frames for calibration
        label = "frames for calibration: " + str(frame_counter) + " / 30"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        gap = 5
        color_bg = (0, 0, 0)
        color_text= (255, 255, 255)
        start = (0, 0)
        (width, height), baseline = cv2.getTextSize(label, font, fontScale, 1)
        label_height = height + baseline + gap
        label_width = width + gap*2
        start_text = (gap, label_height)
        end = (label_width, label_height+gap*2)
        cv2.rectangle(img, start, end, color_bg, -1)
        cv2.putText(img, label, start_text, font, 
            fontScale, color_text, 1, cv2.LINE_AA)
