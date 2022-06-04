import os
import time
import logging
import datetime
import cv2

def get_all_haarcascades():
    HAARCASCADES = {}
    for c in os.listdir(cv2.data.haarcascades):
        if c.endswith('.xml'):
            k = c.split('.')[0].replace('haarcascade_', '')
            v = os.path.join(cv2.data.haarcascades, c)
            HAARCASCADES[k] = v
    return HAARCASCADES

HAARCASCADES = get_all_haarcascades()
COLOURS = {'blue': (255,0,0), 'green': (0,255,0), 'red': (0,0,255)}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File functions
def timestamp(tfmt: str = None) -> str:
    now = datetime.datetime.utcnow()

    if tfmt is None:
        tfmt = "%Y-%m-%dT%H%M%S"

    return datetime.datetime.strftime(now, tfmt)

def make_filename(ext: str = None, suff: str = None, tfmt: str = None) -> str:
    # makes a timestamped filename with specific extension
    fname = timestamp(tfmt)

    if suff is not None:
        fname = '_'.join([timestamp(tfmt), suff])

    if ext is not None:
        fname = '.'.join([fname, ext])

    return fname

def get_frame_dims(capture):
    return int(capture.get(3)), int(capture.get(4))

def rect(frame, obj, lw=2, clr=None):
    # Draw rectangle
    clr = COLOURS.get(clr, (0, 255, 0))
    for (x, y, w, h) in obj:
        cv2.rectangle(frame, (x,y), (x+w, y+h), clr, lw)

def detected(obj):
    return len(obj) > 0

def detected_any(*objs):
    return any([detected(obj) for obj in objs])

def detect(scale=1.1, neighbours=3, rec_buffer=5, out_ext='mp4', out_path='.'):
    recording = False
    detecting = False
    buffer_started_at = None
    within_buffer = False
    
    cap = cv2.VideoCapture(0)
    frame_size = get_frame_dims(cap)
    fourcc = cv2.VideoWriter_fourcc(*out_ext + 'v')
    fname = os.path.join(out_path, make_filename(out_ext))
    face_cascade = cv2.CascadeClassifier(HAARCASCADES['frontalface_default'])

    logging.info("pending")

    while True:
        ret, frame = cap.read()
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            image=greyscale,
            scaleFactor=scale,
            minNeighbors=neighbours,
        )

        rect(frame, obj=faces)

        if detected_any(faces):
            if not recording:
                # New detection - start new video.
                out = cv2.VideoWriter(fname, fourcc, 20, frame_size)
                logging.info("detected 1")
                logging.info("recording on")
            elif not detecting:
                # Started detecting again (previously stopped)
                logging.info("detected 1")

            recording = True
            detecting = True
            within_buffer = False

        elif recording:
            # Still recording, but not detected this time.
            if within_buffer:
                # Check to see if exceeding buffer.
                if (time.time() - buffer_started_at) >= rec_buffer:
                    # Buffer time exceeded. Reset everything & stop recording.
                    recording = False
                    within_buffer = False
                    out.release()
                    logging.info("recording off")
            else:
                # Start buffer timer to wait for new detection event.
                buffer_started_at = time.time()
                within_buffer = True
                logging.info("detected 0")

            detecting = False

        if recording:
            # Write frames to output.
            out.write(frame)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            logging.info("quitting on request")
            break

    out.release() # Release & save video.
    cap.release()
    cv2.destroyAllWindows()
    logging.info("stopped recording")

if __name__ == '__main__':
    rec_path = os.path.join(os.getcwd(), 'rec')
    detect(out_path=rec_path)
