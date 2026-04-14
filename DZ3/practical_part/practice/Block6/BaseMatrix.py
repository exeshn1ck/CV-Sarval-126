import numpy as np
import cv2
import glob
import uuid


class Base():
    def __init__(self):
        pass

    def get_board(self):
        return {}

    def calibrate_by_images(self, source, calib_source, type):
        images = glob.glob(calib_source + "*.jpg") + glob.glob(calib_source + "*.jpeg")
        if len(images):
            img = cv2.imread(images[0])
            calibration_result = self._calibrate(img)
            calibration_result['type'] = type
            calibration_result['source'] = source
            calibration_result['calibration_source'] = calib_source
            return calibration_result
        return None

    def calibrate_by_video(self, source, calib_source, type, all_frames=False):
        cam = calib_source
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        flag, img = cap.read()
        if flag:
            calibration_result = self._calibrate(img)
            calibration_result['type'] = type
            calibration_result['source'] = source
            calibration_result['calibration_source'] = calib_source
            return calibration_result
        return None

    def _calibrate(self, img):
        focal_length = img.shape
        center = (focal_length[1]/2, focal_length[0]/2)
        camera_matrix = np.array(
                                [[focal_length[1], 0, center[0]],
                                [0, focal_length[0], center[1]],
                                [0, 0, 1]], dtype = "double"
                                )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        rvecs = []
        tvecs = []
        resolution = {"w": focal_length[0], "h": focal_length[1]}

        print("-"*20)
        print("Camera Matrix: ", camera_matrix)
        print("Distortion Coefficients : ", dist_coeffs)
        print("Rotation Vectors: ", rvecs)
        print("Translation Vectors: ", tvecs)
        print("-"*20)
        optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coeffs,
                (resolution['w'], resolution['h']),
                1,
                (resolution['w'], resolution['h']),
        )
        return {
            "id": str(uuid.uuid4()),
            "camera_matrix": camera_matrix.tolist(),
            "optimal_camera_matrix": optimal_camera_matrix.tolist(),
            "roi": roi,
            "distortion": dist_coeffs.tolist(),
            "rvecs": [vec.tolist() for vec in rvecs],
            "tvecs": [vec.tolist() for vec in tvecs],
            "resolution": resolution
        }
