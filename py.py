import cv2
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from time import time


####################
##### SETTINGS #####
####################


log_time = True

show_orig = True

SRC = "../vid/2022-10-10_14-47-48_720p_15fps_3600s_compressed.mp4"
FPS = 15

PREV_FRAMES_LEN = 2

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


############################
##### GLOBAL VARIABLES #####
############################


PREV_FRAMES = []

INPUT_TRANSFORM_MATRIX = None


##########################
##### MISC FUNCTIONS ##### 
##########################


def tinit():
    if not log_time:
        return
    global TIME
    TIME = time()


def tlog(msg):
    if not log_time:
        return
    global TIME
    t = time()
    print("%.3fms" % (1000 * (t - TIME)), msg)
    TIME = t


def dist(p1, p2):
    return np.sqrt(sum(pow(abs(p1 - p2), 2)))


def translate(m, x, y):
    [x, y, t] = m @ [x, y, 1]
    return [x / t, y / t]


##########################
##### MAIN FUNCTIONS #####
##########################


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
    global INPUT_TRANSFORM_MATRIX

    # Get perspective transform matrix
    points_1, points_2 = getPerspectiveTransformPoints()
    perspective_transform_matrix = cv2.getPerspectiveTransform(points_1, points_2)

    # Get matrix to move the image so the top left corner of interest area is
    # in the top left frame corner
    interest_area_translated = [translate(perspective_transform_matrix, p[0], p[1]) for p in INTEREST_AREA]
    x_min = min([p[0] for p in interest_area_translated])
    y_min = min([p[1] for p in interest_area_translated])
    interest_area_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]])

    # TODO Rotate the frame so top left and top right have the same y
    #  rotation_fix_matrix = 

    # Multiply in the opposite order!
    matrix = interest_area_matrix
    matrix = matrix @ perspective_transform_matrix

    INPUT_TRANSFORM_MATRIX = np.float32(matrix)


def transformInput(frame):

    w = frame.shape[1]
    h = frame.shape[0]

    matrix = INPUT_TRANSFORM_MATRIX

    # Get size of the future frame
    corners_points_translated = [translate(matrix, *p) for p in INTEREST_AREA]

    max_x = int(max(p[0] for p in corners_points_translated))
    max_y = int(max(p[1] for p in corners_points_translated))

    new_frame = cv2.warpPerspective(frame, matrix, (max_x, max_y))

    return new_frame



async def handleFrame(frame):
    global PREV_FRAMES
    print("================================================================================")

    # To grayscale
    tinit()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tlog("To grayscale")

    # Transform - perspective transform, move and crop to interest area
    transformed = transformInput(gray)
    tlog("Transform")

    if len(PREV_FRAMES) < PREV_FRAMES_LEN:
        # If there are not yet enough frames saved, do nothing (but save frame)
        PREV_FRAMES.append(transformed)
        return transformed

    # transformed = cv2.Laplacian(transformed, cv2.CV_16S, ksize=9).astype(np.uint8)
    # transformed = cv2.medianBlur(transformed, 5)
    # tlog("Median blur")

    diffs = []
    for i in range(PREV_FRAMES_LEN):
        if i != PREV_FRAMES_LEN - 1:
            diffs.append(cv2.subtract(PREV_FRAMES[i], PREV_FRAMES[i + 1]))
        else:
            diffs.append(cv2.subtract(PREV_FRAMES[i], transformed))

    tlog("Diffs")

    # Pop the oldest frame and save this one
    PREV_FRAMES.pop(0)
    PREV_FRAMES.append(transformed)

    #  for i in range(len(diffs)):
        #  diffs[i] = cv2.normalize(diffs[i], None, 0, 255, cv2.NORM_MINMAX)

    # TODO needed?
    diffs_thresholded = []
    for i in range(len(diffs)):
        diffs_thresholded.append(cv2.threshold(diffs[i], 32, 255, cv2.THRESH_BINARY)[1])
    tlog("Threshold")

    frame_or = diffs_thresholded[0]
    for i in range(len(diffs_thresholded) - 1):
        frame_or = cv2.bitwise_or(frame_or, diffs_thresholded[i + 1])
    tlog("or")

    frame_and = diffs_thresholded[0]
    for i in range(len(diffs_thresholded) - 1):
        frame_and = cv2.bitwise_and(frame_and, diffs_thresholded[i + 1])
    tlog("and")

    contours, _ = cv2.findContours(frame_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Erase small contours
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            cv2.fillPoly(frame_or, pts=[c], color=0)
    tlog("contours")

    # morph = cv2.morphologyEx(frame_or, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49,49)))
    morph = cv2.morphologyEx(frame_or, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
    tlog("morphology")

    # arr = []
    # for x,y,w,h in contours:
    #     arr.append((x,y))
    #     arr.append((x+w,y+h))

    # box = cv2.minAreaRect(np.asarray(arr))
    # pts = cv2.boxPoints(box) # 4 outer corners
    # print(pts)
    
    cv2.imshow("last_diff", diffs[-1])

    return morph

    return frame_and
    return frame_or

    #  return dilated

    #  return frame_or.astype(np.uint16)
    sobel = cv2.Sobel(frame_or.astype(np.uint16), ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)
    tlog("Sobel")

    kernel = np.ones((29, 29), np.uint8)
    dilated = cv2.dilate(sobel, kernel, iterations=1)
    tlog("Dilatation")

    return dilated
    #  return cv2.Canny(dilated, 0, 255)

    return frame_and
    return frame_or
    (contours, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    output = frame.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        points = [[x, y], [x + w, y + h]]
        points_translated = [translate(np.linalg.inv(INPUT_TRANSFORM_MATRIX), p[0], p[1]) for p in points]
        points_translated = np.array(points_translated, dtype=np.int)
        cv2.rectangle(output, points_translated[0], points_translated[1], (0, 255, 0), 1)


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

    last_ts = time()

    while video.isOpened():

        ret, frame = video.read()

        if show_orig:
            cv2.imshow("orig", frame)

        output_frame = await handleFrame(frame)
        cv2.imshow("frame", output_frame)

        last_ts += 1 / FPS
        # await asyncio.sleep(1 / FPS - (time() - last_ts))

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
