import cv2
import asyncio
import numpy as np
import time
import matplotlib.pyplot as plt
from timeit import timeit

SRC = "./vid/2022-10-10_14-47-48_720p_15fps_3600s_compressed.mp4"
FPS = 15

OLD_FRAME = None

CAR_CASCADE = cv2.CascadeClassifier('cars.xml')

PERSPECTIVE_TRANSFORM_MATRIX = None

P1 = np.array([380, 650])
P2 = np.array([520, 708])
P3 = np.array([720, 465])
P4 = np.array([625, 435])
P12 = (P1 + P2) // 2
P34 = (P3 + P4) // 2
P1_P2_d = 7
P3_P4_d = 7
P12_P34_d = 16.5

INTEREST_AREA = [
        [306, 700],
        [508, 715],
        [866, 260],
        [700, 250]]



def getPerspectiveTransformMatrix():
    global PERSPECTIVE_TRANSFORM_MATRIX

    def dist(p1, p2):
        return np.sqrt(sum(pow(abs(p1 - p2), 2)))

    # Calculate pixels per meter based on P1_P2_d
    pixels_per_m = dist(P1, P2) / P1_P2_d 

    # Calculate, by how much should we lengthen the P12-P34 line
    p12_p34_pix_d = dist(P12, P34)
    p12_p34_req_pix_d = P12_P34_d * pixels_per_m

    # Move the P34 point (make it longer)
    p12_p34_v = P34 - P12
    p12_p34_req_v = p12_p34_v * (p12_p34_req_pix_d / p12_p34_pix_d)
    p34_req_pix = p12_p34_req_v + P12


    # Move P3 point with P34 being an anchor
    p34_p3_pix_d = dist(P34, P3)
    p34_p4_pix_d = p34_p3_pix_d 

    p34_p3_req_pix_d = (P3_P4_d / 2) * pixels_per_m
    p34_p4_req_pix_d = p34_p3_req_pix_d

    # Lengthen P34-P3
    p34_p3_v = P3 - P34
    p34_p3_req_v = p34_p3_v * (p34_p3_req_pix_d / p34_p3_pix_d)
    p3_req_pix = p34_p3_req_v + P34

    # Lengthen P34-P4
    p34_p4_v = P4 - P34
    p34_p4_req_v = p34_p4_v * (p34_p4_req_pix_d / p34_p4_pix_d)
    p4_req_pix = p34_p4_req_v + P34

    # Move P3 and P4 along the P12-P34 line
    p34_diff_v = p34_req_pix - P34 # Vector from old P34 to new P34
    p3_req_pix += p34_diff_v
    p4_req_pix += p34_diff_v


    #  cv2.line(frame, P1, P2, (255, 0, 0), 2)
    #  cv2.line(frame, P3, P4, (255, 0, 0), 2)
    #  cv2.line(frame, P12, P34, (255, 0, 0), 2)

    #  cv2.line(frame, P12, p34_req_pix.astype(int), (0, 0, 255), 1)

    #  cv2.line(frame, p34_req_pix.astype(int), p3_req_pix.astype(int), (0, 0, 255), 1)
    #  cv2.line(frame, p34_req_pix.astype(int), p4_req_pix.astype(int), (0, 0, 255), 1)

    p3_req_pix[0] -= 7

    #  p4_req_pix[1] += 10

    # Get four previous and next points: P1, P2, P3 and P4
    points_1 = np.array([P1, P2, P3, P4], np.float32)
    points_2 = np.array([P1, P2, p3_req_pix.astype(int), p4_req_pix.astype(int)], np.float32)
    #  points_2 = np.array([P1, P2, p3_req_pix.astype(int), P4], np.float32)
    #  points_2 = np.array([[0, 719], [1279, 719], [1279, 0], [0, 0]], np.float32)
    print(points_1)
    print(points_2)

    # Get transform matrix from those sets of points
    PERSPECTIVE_TRANSFORM_MATRIX = cv2.getPerspectiveTransform(points_1, points_2)
    #  PERSPECTIVE_TRANSFORM_MATRIX = np.float32([[1, 0, 10], [0, 1, 10]])
    print(PERSPECTIVE_TRANSFORM_MATRIX)



def perspectiveTransform(frame):

    return frame
    matrix = PERSPECTIVE_TRANSFORM_MATRIX
    print(matrix)


    def translate(m, x, y):
        #  [x, y] = np.dot(m, [x, y, 1])
        #  return [x, y]

        #  [x, y, t] = np.dot(m, [x, y, 1])
        [x, y, t] = m.dot([x, y, 1])
        #  print(x, y, t)
        return [x / t, y / t]

    print("0 0: ", translate(matrix, 0, 0))
    print("1280 0: ", translate(matrix, 1280, 0))
    print("1280 720: ", translate(matrix, 1280, 720))
    print("0 720: ", translate(matrix, 0, 720))

    #  print("")

    #  print("0 0: ", translate(matrix, 0, 0))
    #  print("0 1280: ", translate(matrix, 0, 1280))
    #  print("720 1280: ", translate(matrix, 720, 1280))
    #  print("720 0: ", translate(matrix, 720, 0))

    #  return None

    #  while True:
        #  print("Input:")
        #  x = input()
        #  y = input()
        #  print(translate(matrix, float(x), float(y)))

    # Find where the corners translate to TODO nefunguje
    #  corners_points = np.array([[0, 0, 1], [h - 1, 0, 1], [h - 1, w - 1, 1], [0, w - 1, 1]], np.float32);
    corners_points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], np.float32);
    corners_points_translated = [translate(matrix, *corners_points[i]) for i in range(4)]

    #  matrix_inv = np.linalg.inv(matrix)

    #  print(translate(matrix_inv, 0, 0))

    print("===")
    print(corners_points)
    print(corners_points_translated)

    # Get inverse matrix and apply it to the left top corner or something


    max_x = int(max([point[0] for point in corners_points_translated]))
    max_y = int(max([point[1] for point in corners_points_translated]))

    x_0 = 000
    y_0 = 000

    h_tmp = max(max_y, h)
    w_tmp = max(max_x, w)
    #  print(w_tmp)
    #  print(h_tmp)


    #  frame_tmp = np.zeros((h_tmp, w_tmp, 3), dtype=np.uint8)
    frame_tmp = np.zeros((10000, 10000), dtype=np.uint8)

    frame_tmp[y_0: y_0 + h, x_0: x_0 + w] = frame

    #  frame_tmp[219: 219 + h, 261: 261 + w] = frame

    #  frame_tmp[400: 400+ h, 261: 261 + w] = frame

    xs = []
    ys = []

    #  inputs = [720 * [0] + list(range(1280)) + 720 * [1280] + list(range(1280)), \
              #  list(range(720)) + 1280 * [720] + list(range(720)) + 1280 * [0]]

    #  inputs = np.array(inputs)


    #  vectors = []
    #  for i in range(inputs.shape[1]):
        #  x, y = inputs[0][i], inputs[1][i]
        #  output = translate(matrix, x, y)
        #  u = output[0] - x
        #  v = output[1] - y
        #  vector = [x, y, u, v]
        #  vectors.append(vector)

    #  vectors = np.array(vectors)

    #  vectors = vectors.reshape(vectors.shape[1], vectors.shape[0])

    #  plt.quiver(*vectors)
    #  plt.show()

    #  print(matrix)   
    #  return frame

    #  tmp1 = range(1280)
    #  top = np.array([translate(matrix, x, 0) for x in tmp1])

    #  tmp = top.reshape((top.shape[1], top.shape[0]))
    #  print(top.shape)
    #  plt.scatter(tmp[0], tmp[1])
    #  plt.show()

    #  left = np.array([translate(matrix, 0, y) for y in range(720)])
    #  tmp = left.reshape((left.shape[1], left.shape[0]))
    #  print(left.shape)
    #  plt.scatter(tmp[0], tmp[1])
    #  plt.show()


    def a(img, M, dsize):
        mtr = img
        R,C = dsize
        dst = np.zeros((R,C))
        for i in range(mtr.shape[0]):
            for j in range(mtr.shape[1]):
                res = np.dot(M, [i,j,1])
                i2,j2,_ = (res / res[2] + 0.5).astype(int)
                if i2 >= 0 and i2 < R:
                    if j2 >= 0 and j2 < C:
                        dst[i2,j2] = mtr[i,j]
        return dst

    #  new_frame = a(frame, matrix, (frame.shape[0], frame.shape[1]))


    #  new_frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
    #  new_frame = cv2.warpPerspective(frame_tmp, matrix, (h_tmp, w_tmp))
    new_frame = cv2.warpPerspective(frame_tmp, matrix, (10000, 950))
    #  new_frame = cv2.warpPerspective(frame, matrix, (10000, 1000))

    #  return frame
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

    # TODO Crop to interest area (rectangle, discarded pixels in it are black)


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
