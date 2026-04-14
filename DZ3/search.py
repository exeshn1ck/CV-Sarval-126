import cv2

img = cv2.imread(r"D:/YandexDisk/MAI/2nd term/CV/DZ3/practical_part/completed/calibration_data/1.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sizes_to_test = [
    (7, 5), (5, 7),
    (8, 5), (5, 8),
    (8, 6), (6, 8),
    (9, 6), (6, 9),
    (9, 5), (5, 9)
]

for size in sizes_to_test:
    found, corners = cv2.findChessboardCorners(gray, size, None)
    print(size, "->", found)