import os
from time import sleep
import glob
import threading
from signal import pthread_kill, SIGINT
from datetime import datetime
from pprint import pprint
import ffmpeg
import math


STOP_FILEPATH = "stop"

MOUNTPOINT = "traffic_cams"
REMOTE_PATH = "vutbrdrive:traffic_cams"

#  WHITELIST = ["breclav"]
#  WHITELIST = ["breclav_post"]
WHITELIST = []

COMPRESSION_SEGMENT_DURATION = 600 # In seconds. Ignored if -1

MAX_SIZE = 1_000_000_000 # Max size of item which can be compressed in bytes - bigger will be ignored
#  MAX_SIZE = 999_999_999_999


def out(*args):
    print(f"{timestamp()}: ", end="")
    print(*args)


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def mount():
    cmd = f"rclone mount {REMOTE_PATH} {MOUNTPOINT} --vfs-cache-mode=full"
    out(f"Executing: {cmd}")
    os.system(cmd)


def compress(f1, f2):

    file_size = os.stat(f1).st_size # In bytes

    # Get duration
    if "s." in f1:
        file_duration = int(f1.split("s.")[0].split("_")[-1])
    else: # "s_" in f1
        file_duration = int(f1.split("s_")[0].split("_")[-1])

    segment = 0

    try:

        # Split the file into segments of COMPRESSION_SEGMENT_DURATION while
        # also compressing those segments
        while True:

            out(f"Creating a compressed segment (of index) {segment} of length {COMPRESSION_SEGMENT_DURATION}s")
            start_timestamp = segment * (COMPRESSION_SEGMENT_DURATION)

            end_timestamp = start_timestamp + COMPRESSION_SEGMENT_DURATION
            if end_timestamp > file_duration:
                end_timestamp = file_duration

            stream = ffmpeg.input(f1)
            stream = ffmpeg.output(stream, f"segment_{segment}_compressed.mp4", vsync="2", vcodec="libx265", preset="veryfast", threads=1, ss=start_timestamp, to=end_timestamp)
            stream = stream.global_args("-stats", "-loglevel", "warning")
            stream.run()

            if end_timestamp == file_duration:
                break

            segment += 1
            print("")
            out(f"Segment (of index) {segment} created and compressed")

        # Write names of segment files to a .txt file
        with open("files.txt", "w") as segment_file:
            for s in range(segment + 1):
                segment_file.write(f"file segment_{s}_compressed.mp4\n")

        # Concat all segments
        out(f"Concatenating {segment + 1} segments")
        stream = ffmpeg.input("files.txt", f="concat", safe=0)
        stream = ffmpeg.output(stream, f2, c="copy")
        stream = stream.global_args("-stats", "-loglevel", "warning")
        stream.run()
        out(f"Concatenated {segment + 1} segments successfully")

    except Exception as e:
        raise

    finally:
        for s in range(segment + 1):
            name = f"segment_{s}_compressed.mp4"
            if os.path.exists(name):
                os.remove(name)
        if os.path.exists("files.txt"):
            os.remove("files.txt")


def main():

    if not os.path.isdir(MOUNTPOINT):
        os.mkdir(MOUNTPOINT)

    out(f"Mounting {REMOTE_PATH} to {MOUNTPOINT}")

    try:
        t = threading.Thread(target=mount, args=(), daemon=True)
        t.start()

        sleep(5)

        paths = []
        compressed_paths = []
        for f in glob.glob(f"./{MOUNTPOINT}/**", recursive=True):

            # Ignore compressed files but save their names for later
            if "compressed" in f:
                f = os.path.relpath(f, MOUNTPOINT)
                compressed_paths.append(f)
                continue

            # Ignore non .mp4 or .ts files
            if not (f.endswith(".mp4") or f.endswith(".ts")):
                continue

            # Ignore whitelisted files
            whitelisted = False
            for whitelist_item in WHITELIST:
                if whitelist_item in f:
                    whitelisted = True
                    break
            if whitelisted:
                continue

            # If the file is too big, ignore it
            size = os.stat(f).st_size # In bytes
            if size > MAX_SIZE:
                continue

            f = os.path.relpath(f, MOUNTPOINT)
            paths.append(f)

        # Ignore all files that also exist as compressed
        for path in paths.copy():
            compressed_name = os.path.splitext(path)[0] + "_compressed.mp4"
            if compressed_name in compressed_paths:
                paths.remove(path)

        pprint(paths)

        # Kill the mount thread
        if t.is_alive():
            try:
                out(f"Killing mount thread")
                pthread_kill(t.ident, SIGINT)
            except:
                pass

    except KeyboardInterrupt:
        out("Stop requested by keyboard interrupt")

    except Exception as e:
        out(f"Unknown exception when reading drive contents: {e}")

    finally:
        out("To be sure, unmounting using fusermount")
        cmd = f"fusermount -uz {MOUNTPOINT}"
        out(f"Executing: {cmd}")
        os.system(cmd)

    for path in paths:
        try:
            print("================================================================================")

            if os.path.exists(STOP_FILEPATH):
                out("Stop requested. Exiting")
                return

            out(f"Processing path {path}")

            name = os.path.basename(path)
            new_name = os.path.splitext(name)[0] + "_compressed.mp4"

            cmd = f"rclone copy {REMOTE_PATH}/{path} ./"
            out(f"Executing: {cmd}")
            os.system(cmd)

            if not os.path.exists(name):
                raise Exception(f"Could not download {REMOTE_PATH}/{path}. Ignoring")

            out(f"Starting compression: {name} => {new_name}")
            compress(name, new_name)

            if not os.path.exists(new_name):
                raise Exception(f"Compressed file {new_name} not found. Ignoring")

            # Upload to drive
            dirpath = os.path.dirname(path)
            cmd = f"rclone copy {new_name} {REMOTE_PATH}/{dirpath}"
            out(f"Executing: {cmd}")
            os.system(cmd)

            # Remove uncompressed from drive
            cmd = f"rclone delete {REMOTE_PATH}/{dirpath}/{name}"
            out(f"Executing: {cmd}")
            os.system(cmd)

        except KeyboardInterrupt:
            out(f"Stop requested when when compressing {path}")
            return

        except Exception as e:
            out(f"Exception when compressing {path}: {e}")

        finally:
            if os.path.exists(name):
                out(f"Removing {name}")
                os.remove(name)

            if os.path.exists(new_name):
                out(f"Removing {new_name}")
                os.remove(new_name)


if __name__ == "__main__":
    try:
        while not os.path.exists(STOP_FILEPATH):
            main()
    except Exception as e:
        out(f"Exception: {str(e)}")
