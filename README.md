# detector

Basic face detection script.

Starts the camera and waits to detect a face. If a face is detected, it starts recording. If the face stops being detected it continues recording until a recording buffer duration has been exceeded, after which the recording stops but monitoring continues. If a face is detected again, it restarts recording.

Run with: `$ python detector.py`
