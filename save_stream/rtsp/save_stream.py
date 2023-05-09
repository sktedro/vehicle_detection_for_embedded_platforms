from time import time
from datetime import datetime
import threading
import cv2
import os


# Stop after 10 minutes
duration_s = 60 * 60

fps = 25

URL = f"rtsp://user:pass@..." # TODO

os.system(f"mkdir -p to_process")
os.system(f"mkdir -p processed")


if __name__ == '__main__':

    print(f"Saving next {duration_s}s of stream")

    start_timestamp = time()
    start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print(f"Starting: {start_time} ({start_timestamp})")

    capture = cv2.VideoCapture(URL)

    if (capture.isOpened() == False):
        raise Exception("Error reading video")

    w = int(capture.get(3))
    h = int(capture.get(4))
    size = (w, h)

    filename = f"{start_time}_{h}p_{fps}fps_{duration_s}s.mp4"

    output = cv2.VideoWriter(filename,
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, size)

    stop_timestamp = start_timestamp + duration_s

    while capture.isOpened() and round(time()) < stop_timestamp:
        ret, frame = capture.read()
        output.write(frame)
        #  thread = threading.Thread(target=output.write, args=(frame,))
        #  thread.start()

    capture.release()
    output.release()

    print(f"Stopping: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} ({time()})")
    print(f"Length: {round(time() - start_timestamp, 3)}s")

    # Get real length:
    video = cv2.VideoCapture(filename)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    duration = frame_count / fps
    print(f"File length: {round(duration, 3)}s")

    os.system(f"mv {filename} ./to_process/")
    print(f"Saved to ./to_process/{filename}")
    print()
