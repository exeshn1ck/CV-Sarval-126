import cv2
import numpy as np
import math
import sys
import json
import argparse
import os
from datetime import datetime

from ChessBoard import ChessBoard

mouse_position = [0,0]
temp_point = None
chess_axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,-0.1]]).reshape(-1,3)


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None

def scale_matrix_2d(vec):
    x, y = vec[:2]
    return np.matrix([[x,0,0],
                      [0,y,0],
                      [0,0,1]])

def translation_matrix_2d(vec):
    x, y = vec[:2]
    return np.matrix([[1,0,x],
                      [0,1,y],
                      [0,0,1]])

def translation_matrix_3d(vec):
    x, y, z = vec[:3]
    return np.matrix([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]])

def rotation_matrix_3d(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction."""
    angle = math.radians(angle)
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def find_board(img, chess_board):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, tuple(chess_board.pattern_size), None)
    if ret and len(corners) > 0:
        cv2.drawChessboardCorners(img, tuple(chess_board.pattern_size), corners, ret)
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), chess_board.criteria)
        return corners
    else:
        None

def draw_axis(img, corners, imgpts):
    corner = tuple([ int(coord) for coord in corners[0].ravel()] )
    p1 = tuple([ int(coord) for coord in imgpts[0].ravel()] )
    p2 = tuple([ int(coord) for coord in imgpts[1].ravel()] )
    p3 = tuple([ int(coord) for coord in imgpts[2].ravel()] )
    img = cv2.line(img, corner, p1, (255,0,0), 5) # x - blue
    img = cv2.line(img, corner, p2, (0,255,0), 5) # y - green
    img = cv2.line(img, corner, p3, (0,0,255), 5) # z - red
    return img

def draw_points(img, points, color, alpha=1, radius=5):
    for point in points:
        x, y = point[:2]
        overlay = img.copy()
        cv2.circle(overlay, (int(x), int(y)), radius, color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def draw_label(img, position):
    label = "x: %s | y: %s | z: %s "%(position[0], position[1], position[2])
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

def get_local_to_world_matrix(camera_world_position, board_rotation):
    board_world_rotation = rotation_matrix_3d(board_rotation, [0,0,1])
    camera_rot_pos = np.append(camera_world_position, [1])
    camera_rot_pos = board_world_rotation @ camera_rot_pos
    board_world_translation = translation_matrix_3d(-1 * camera_rot_pos)
    transform = board_world_translation @ board_world_rotation
    return transform

def screen_to_camera(point, camera_matrix, rotM):
    i_rot = np.linalg.inv(rotM)
    c_x = camera_matrix[0][2]
    c_y = camera_matrix[1][2]
    f_x = camera_matrix[0][0]
    f_y = camera_matrix[1][1]
    u = point[0] - c_x
    v = point[1] - c_y
    z_c = 1
    x = z_c * (u / f_x)
    y = z_c * (v / f_y)
    camera_point = np.matrix([x, y, z_c]).T
    world_point = np.matmul(i_rot, camera_point)
    return world_point


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--source', "-s", default=0, help="Camera source")
    parser.add_argument('--calibration', "-c", default=None, help="Path to calibration file")
    args = parser.parse_args()

    # create cfg
    cfg = {
        "camera": {
            'source': int(args.source) if len(args.source)==1 else args.source,
            "calibration_path": args.calibration,
        },
        "board": {}
    }

    # read calibration file
    with open(cfg['camera']["calibration_path"], 'r') as json_file:
        calibration_meta = json.load(json_file)
        cam_calibration = None
        if 'cameras' in calibration_meta and len(calibration_meta['cameras']):
            for camera in calibration_meta['cameras']:
                cam_calibration = camera
        if 'board' in calibration_meta:
            cfg['board']["pattern_size"] = calibration_meta['board']["pattern_size"]
            cfg['board']["square_size"] = calibration_meta['board']["square_size"]
    if cam_calibration is None:
        print("No calibration for camera")
        sys.exit(0)
    cfg["camera"]['calibration'] = cam_calibration

    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(cfg["camera"]['source'])

    chess_board = ChessBoard(
            pattern_size = cfg['board']['pattern_size'],
            square_size = cfg['board']['square_size']
        )

    camera_matrix = np.array(cfg['camera']['calibration']['camera_matrix'])
    dist_coeffs = np.array(cfg['camera']['calibration']['distortion'])

    frame_dim = (640,480)

    while True:
        flag, img = cap.read()
        if not flag:
            break

        # find and draw chessboard
        corners = find_board(img, chess_board)

        # find camera and board pos
        if corners is not None:
            ret, rvecs, tvecs = cv2.solvePnP(chess_board.objp, corners, camera_matrix, dist_coeffs)
            rotM = cv2.Rodrigues(rvecs)[0]
            camera_world_position = -np.matrix(rotM).T * np.matrix(tvecs)

            # поворот доски по оси z относительно камеры
            angle = np.degrees(rvecs[2])

            # позиция камеры относительно 0,0 базового объекта
            camera_world_position = np.ravel(camera_world_position)

            # draw board axis
            imgpts, jac = cv2.projectPoints(chess_axis, rvecs, tvecs, camera_matrix, dist_coeffs)
            img_chess = draw_axis(img, corners, imgpts)

            draw_label(img, camera_world_position)

        img = cv2.resize(img, frame_dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('frame', img)
        ch = cv2.waitKey(1)
        if ch == ord('q') or ch == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
