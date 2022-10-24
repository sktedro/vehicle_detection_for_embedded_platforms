import cv2
import asyncio
import numpy as np
import time
import matplotlib.pyplot as plt
from timeit import timeit

SRC = "../vid/2022-10-10_14-47-48_720p_15fps_3600s_compressed.mp4"
FPS = 15

OLD_FRAME = None

PERSPECTIVE_TRANSFORM_MATRIX = None

PERSPECTIVE_POINTS = np.array([
    [380, 650],
    [520, 708],
    [720, 465],
    [625, 435]])
P1_P2_d = 7
P3_P4_d = 7
P12_P34_d = 16.5

INTEREST_AREA = [
        [300, 700],
        [500, 719],
        [900, 250],
        [750, 250]]


def dist(p1, p2):
    return np.sqrt(sum(pow(abs(p1 - p2), 2)))

def translate(m, x, y):
    [x, y, t] = m.dot([x, y, 1])
    return [x / t, y / t]

def getPerspectiveTransformPoints():
    P12 = (PERSPECTIVE_POINTS[0] + PERSPECTIVE_POINTS[1]) // 2
    P34 = (PERSPECTIVE_POINTS[2] + PERSPECTIVE_POINTS[3]) // 2

    # Calculate pixels per meter based on P1_P2_d
    pixels_per_m = dist(PERSPECTIVE_POINTS[0], PERSPECTIVE_POINTS[1]) / P1_P2_d 

    # Calculate, by how much should we lengthen the P12-P34 line
    p12_p34_pix_d = dist(P12, P34)
    p12_p34_req_pix_d = P12_P34_d * pixels_per_m

    # Move the P34 point (make it longer)
    p12_p34_v = P34 - P12
    p12_p34_req_v = p12_p34_v * (p12_p34_req_pix_d / p12_p34_pix_d)
    p34_req_pix = p12_p34_req_v + P12

    # Move P3 point with P34 being an anchor
    p34_p3_pix_d = dist(P34, PERSPECTIVE_POINTS[2])
    p34_p4_pix_d = p34_p3_pix_d 

    p34_p3_req_pix_d = (P3_P4_d / 2) * pixels_per_m
    p34_p4_req_pix_d = p34_p3_req_pix_d

    # Lengthen P34-P3
    p34_p3_v = PERSPECTIVE_POINTS[2] - P34
    p34_p3_req_v = p34_p3_v * (p34_p3_req_pix_d / p34_p3_pix_d)
    p3_req_pix = p34_p3_req_v + P34

    # Lengthen P34-P4
    p34_p4_v = PERSPECTIVE_POINTS[3] - P34
    p34_p4_req_v = p34_p4_v * (p34_p4_req_pix_d / p34_p4_pix_d)
    p4_req_pix = p34_p4_req_v + P34

    # Move P3 and P4 along the P12-P34 line
    p34_diff_v = p34_req_pix - P34 # Vector from old P34 to new P34
    p3_req_pix += p34_diff_v
    p4_req_pix += p34_diff_v

    # Get four previous and next points: P1, P1, P3 and P4
    points_1 = np.array(PERSPECTIVE_POINTS, np.float32)
    points_2 = np.array([PERSPECTIVE_POINTS[0], PERSPECTIVE_POINTS[1], p3_req_pix.astype(int), p4_req_pix.astype(int)], np.float32)

    return points_1, points_2


def getTransformMatrix():
    global PERSPECTIVE_TRANSFORM_MATRIX

    # Get perspective transform matrix
    points_1, points_2 = getPerspectiveTransformPoints()
    #  points_1 = np.float32([translate(interest_area_matrix, *p) for p in points_1])
    #  points_2 = np.float32([translate(interest_area_matrix, *p) for p in points_2])
    perspective_transform_matrix = cv2.getPerspectiveTransform(points_1, points_2)
    #  print(perspective_transform_matrix)

    # Get matrix to move the image so the top left corner of interest area is
    # in the top left frame corner
    interest_area_translated = [translate(perspective_transform_matrix, p[0], p[1]) for p in INTEREST_AREA]
    x_min = min([p[0] for p in interest_area_translated])
    y_min = min([p[1] for p in interest_area_translated])
    interest_area_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]])
    #  print(interest_area_matrix)

    # TODO Rotate the frame so top left and top right have the same y
    #  x_max = max([p[0] for p in interest_area_translated])
    #  vector_1 = [1, 0]
    #  vector_2 = 
    #  angle = 

    #  rotation_fix_matrix = 

    # Multiply in the opposite order!
    matrix = interest_area_matrix
    matrix = matrix.dot(perspective_transform_matrix)

    PERSPECTIVE_TRANSFORM_MATRIX = np.float32(matrix)



def perspectiveTransform(frame):

    w = frame.shape[1]
    h = frame.shape[0]

    matrix = PERSPECTIVE_TRANSFORM_MATRIX

    # Get size of the future frame
    #  corners_points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.float32);
    #  corners_points_translated = [translate(matrix, *corners_points[i]) for i in range(4)]
    corners_points_translated = [translate(matrix, *p) for p in INTEREST_AREA]

    max_x = int(max(p[0] for p in corners_points_translated))
    max_y = int(max(p[1] for p in corners_points_translated))

    new_frame = cv2.warpPerspective(frame, matrix, (max_x, max_y))

    return new_frame



def smoothen(frame, kernel_side_size, power):
    frame = np.array(frame)

    # Smoothen out noise with a median filter
    kernel = np.ones((kernel_side_size, kernel_side_size), np.float32) / power
    filtered = cv2.filter2D(frame, -1, kernel)

    # TODO Threshold so dark pixels are deleted
    ret, thresholded = cv2.threshold(filtered, 4, 255, cv2.THRESH_TOZERO)

    # TODO Increase brightness of pixels that weren't filtered out
    #  print(thresholded.shape)
    #  hsv = cv2.cvtColor(thresholded, cv2.COLOR_BGR2HSV)
    #  h, s, v = cv2.split(hsv)

    #  multiplier = 255 / max(v)
    #  print(v[0: 10])
    #  v *= multiplier
    #  print(v[0: 10])

    #  final_hsv = cv2.merge((h, s, v))
    #  output = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    #  timeit(lambda: max(255 / max(thresholded.flatten()), 1))
    #  multiplier = 255 / max(thresholded.flatten())
    #  output = thresholded * multiplier
    output = cv2.normalize(thresholded, None, 0, 255, cv2.NORM_MINMAX)

    return output



async def handleFrame(frame):
    global OLD_FRAME

    # To grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perspective transform
    transformed = perspectiveTransform(gray)

    return transformed

    # Subtract the previous frame from the actual frame
    if OLD_FRAME is None:
        OLD_FRAME = transformed
        return transformed
    diff = cv2.subtract(transformed, OLD_FRAME)
    OLD_FRAME = transformed

    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


    #  a = cv2.fastNlMeansDenoising(diff, None, h=10, searchWindowSize=2)
    #  diff = a

    smoothened_1 = cv2.GaussianBlur(diff, (25, 25), 0)
    ret, thresholded = cv2.threshold(smoothened_1, 32, 255, cv2.THRESH_TOZERO)
    thresholded = cv2.normalize(thresholded, None, 0, 255, cv2.NORM_MINMAX)
    #  ret, thresholded = cv2.threshold(smoothened_1, 64, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN)

    #  return thresholded

    (contours, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    output = frame.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        points = [[x, y], [x + w, y + h]]
        points_translated = [translate(np.linalg.inv(PERSPECTIVE_TRANSFORM_MATRIX), p[0], p[1]) for p in points]
        points_translated = np.array(points_translated, dtype=np.int)
        cv2.rectangle(output, points_translated[0], points_translated[1], (0, 255, 0), 1)


    return output

    smoothened_1 = smoothen(output, 3, 32)
    return smoothened_1

    #  smoothened_2 = smoothen(smoothened_1, 32, 128)
    #  return smoothened_2

    #  output = frame

    # Detects cars of different sizes in the input image
    #  cars = CAR_CASCADE.detectMultiScale(output, 1.3, 5)
    # To draw a rectangle in each cars
    #  for (x, y, w, h) in cars:
        #  cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output

async def main():
    global w
    global h
    global FPS

    # TODO Get perspective transformation matrix
    getTransformMatrix()

    video = cv2.VideoCapture(SRC)

    if (video.isOpened() == False):
        raise Exception("Error reading video file")

    w = int(video.get(3))
    h = int(video.get(4))

    last_ts = time.time()

    while video.isOpened():

        ret, frame = video.read()

        output_frame = await handleFrame(frame)
        cv2.imshow("frame", output_frame)

        last_ts += 1 / FPS
        await asyncio.sleep(1 / FPS - (time.time() - last_ts))

        if cv2.waitKey(2) & 0xFF == ord("="):
            FPS *= 2

        if cv2.waitKey(2) & 0xFF == ord("-"):
            FPS /= 2

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
