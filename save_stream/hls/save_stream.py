import os
import threading
import ffmpeg
import subprocess
from pprint import pprint
from datetime import datetime
from time import sleep
from signal import pthread_kill, SIGINT


##################
##### CONFIG #####
##################


DEBUG = False

# Name of the config file
CONFIG_FILEPATH = "config_fl"

# Filepath of a file that stops the recording loop
STOP_FILEPATH = "stop"

# Duration in seconds per file
DURATION_S = 10 * 60


#####################
##### FUNCTIONS #####
#####################


def out(*args):
    print(f"{timestamp()}: ", end="")
    print(*args)


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def getSources():
    sources = []
    with open(CONFIG_FILEPATH) as f:
        for line in f.readlines():
            if line[0] == "#":
                continue
            split = line.replace("\n", "").split(",")
            sources.append((split[0], "".join(split[1:])))

    return sources


# Requires almost no CPU
def download(url, filepath):
    out(f"Executing: ffmpeg -i {url} -t {DURATION_S} -c copy -loglevel warning {filepath}")
    stream = ffmpeg.input(url)
    stream = ffmpeg.output(stream, filepath, t=DURATION_S, c="copy", loglevel="warning")
    stream.run()


def download_and_save(source):
    name, url = source

    # Create directory if it doesn't exist yet
    if not os.path.isdir(name):
        os.mkdir(name)

    # Get filepath
    start_timestamp = timestamp()
    filename = f"{start_timestamp}_{DURATION_S}s.mp4"
    filepath = os.path.join(name, filename)

    t = threading.Thread(target=download, args=(url, filepath,), daemon=True)
    t.start()

    # Sleep while threads are recording
    sleep(DURATION_S)

    try:
        # Wait for up to 10 seconds for the thread, if it is still running
        seconds_waited = 0
        while t.is_alive() and seconds_waited < 10:
            sleep(1)
            seconds_waited += 1

        # If it is still running, kill it with SIGINT so it finishes properly
        if t.is_alive():
            try:
                pthread_kill(t.ident, SIGINT)
                out(f"Killing {name} download")
            except:
                pass

        # Wait for up to 10 seconds for the thread, if it is still running
        seconds_waited = 0
        while t.is_alive() and seconds_waited < 10:
            sleep(1)
            seconds_waited += 1

        # If it is still running, kill it with SIGINT so it doesn't hang for
        # hours (send 3, for total of 4 for hard exit
        if t.is_alive():
            try:
                pthread_kill(t.ident, SIGINT)
                if t.is_alive():
                    pthread_kill(t.ident, SIGINT)
                if t.is_alive():
                    pthread_kill(t.ident, SIGINT)
                out(f"Killing {name} download for hard stop")
            except:
                pass

        # Wait for the thread to finish
        try:
            t.join()
        except KeyboardInterrupt:
            pass

    except Exception as e:
        out(f"Downloading {name} finished with exception: {str(e)}")

    try:
        # Check duration of the file and ignore the file if the duration is too
        # different from the required duration, otherwise, update the filename
        # Give it room of a 2 seconds big error
        try:
            #  duration = round(float(ffmpeg.probe(filepath)["format"]["duration"]))
            if not os.path.exists(filepath):
                raise Exception(f"File {filepath} does not exist")
            cmd = f"ffprobe -loglevel error -show_format {filepath} | sed -n '/duration/s/.*=//p'"
            duration = int(float(subprocess.check_output(cmd, shell=True)))
        except Exception as e:
            out(f"Cannot get duration of {name}: {str(e)}")
            duration = 0
        if duration < DURATION_S - 2 or duration > DURATION_S + 2:
            out(f"Duration mismatch for {name} (have {duration}). Ignoring and deleting")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        out(f"Downloaded {name} successfully")

        new_filename = f"{start_timestamp}_{duration}s.mp4"

        # Move to "done" folder
        if not os.path.isdir("done"):
            os.mkdir("done")
        if not os.path.isdir(f"done/{name}"):
            os.mkdir(f"done/{name}")
        os.system(f"mv {filepath} done/{name}/{new_filename}")

        out(f"Saved {name}")

    except Exception as e:
        out(f"Exception when saving {name}: {str(e)}")


if __name__ == "__main__":
    try:
        while not os.path.exists(STOP_FILEPATH):

            out(f"Starting capturing {DURATION_S}s of streams")

            sources = getSources()
            if DEBUG:
                out(f"Sources:")
                pprint(sources)

            threads = []
            for source in sources:
                threads.append(threading.Thread(target=download_and_save, args=(source,), daemon=True))

            # Start all threads
            for t in threads:
                t.start()

            # Wait for all threads to finish
            for t in threads:
                try:
                    t.join()
                except KeyboardInterrupt:
                    pass

            out(f"All done")

    except Exception:
        raise

    finally:
        os.system('stty sane')
