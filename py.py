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


def getPerspectiveTransformMatrix():
    global PERSPECTIVE_TRANSFORM_MATRIX

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
    #  points_2 = np.array([PERSPECTIVE_POINTS[0], PERSPECTIVE_POINTS[1], p3_req_pix.astype(int), PERSPECTIVE_POINTS[3]], np.float32)
    #  points_2 = np.array([[0, 719], [1279, 719], [1279, 0], [0, 0]], np.float32)
    print(points_1)
    print(points_2)

    # Get transform matrix from those sets of points
    PERSPECTIVE_TRANSFORM_MATRIX = cv2.getPerspectiveTransform(points_1, points_2)
    #  PERSPECTIVE_TRANSFORM_MATRIX = np.float32([[1, 0, 10], [0, 1, 10]])
    print(PERSPECTIVE_TRANSFORM_MATRIX)

    # Move the top left corner to [0, 0] (create a matrix to do that)
    wrong_origin = translate(PERSPECTIVE_TRANSFORM_MATRIX, 0, 0)
    translation = [-i for i in wrong_origin] # Move by the negative value
    translation_matrix = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
            ])

    # TODO Rotate the frame so top left and top right have the same y

    PERSPECTIVE_TRANSFORM_MATRIX = translation_matrix.dot(PERSPECTIVE_TRANSFORM_MATRIX)
    print(PERSPECTIVE_TRANSFORM_MATRIX)



def translatePerspectivePointsToInterestArea():
    global PERSPECTIVE_POINTS
    print(PERSPECTIVE_POINTS)
    x_min = INTEREST_AREA[0][0]
    x_max = INTEREST_AREA[0][0]
    y_min = INTEREST_AREA[0][1]
    y_max = INTEREST_AREA[0][1]
    for p in PERSPECTIVE_POINTS:
        p[0] -= x_min
        p[1] -= y_min
    print(PERSPECTIVE_POINTS)



def cropToInterestArea(frame):
    matrix = PERSPECTIVE_TRANSFORM_MATRIX
    x_min = INTEREST_AREA[0][0]
    x_max = INTEREST_AREA[0][0]
    y_min = INTEREST_AREA[0][1]
    y_max = INTEREST_AREA[0][1]

    for p in INTEREST_AREA:
        x_min = min(x_min, p[0])
        x_max = max(x_max, p[0])
        y_min = min(y_min, p[1])
        y_max = max(y_max, p[1])
    #  print(x_min, x_max, y_min, y_max)

    #  o_w = x_max - x_min
    #  o_h = y_max - y_min
    #  output = np.zeros((o_w, o_h), dtype=np.uint8)
    output = frame[y_min: y_max, x_min: x_max]

    # TODO pixels outside should be black

    return output



def perspectiveTransform(frame):

    w = frame.shape[1]
    h = frame.shape[0]

    matrix = PERSPECTIVE_TRANSFORM_MATRIX

    # Get size of the future frame
    corners_points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.float32);
    corners_points_translated = [translate(matrix, *corners_points[i]) for i in range(4)]
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

    # Crop to interest area (rectangle, discarded pixels in it are black)
    cropped = cropToInterestArea(gray)

    # Perspective transform
    transformed = perspectiveTransform(cropped)

    #  return transformed

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

    return thresholded

    (contours, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    output = frame.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)


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

    translatePerspectivePointsToInterestArea()

    # TODO Get perspective transformation matrix
    getPerspectiveTransformMatrix()

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
