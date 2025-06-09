import cv2
import os
import time

path = '...camera' # Update this to actual path to save the video
# output_file = os.path.join(path, 'output.mp4')
output_file = os.path.join(path, 'test.avi')  # Change to .avi
# change name
frame_per_second = 5

# Open the default camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 10)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set standard resolution
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera")
    exit()

# Get frame size
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_file, fourcc, frame_per_second, (frame_width, frame_height))


# Check if VideoWriter is opened
if not out.isOpened():
    print("Error: VideoWriter not opened!")
    cam.release()
    exit()

print(f"Recording started. Saving video to: {output_file}")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error: Could not read frame")
        break

    out.write(frame) 
    cv2.imshow('Camera', frame)  

    if cv2.waitKey(1) == ord('q'):
        break  

    time.sleep(1/frame_per_second)

cam.release()
out.release()
cv2.destroyAllWindows()


if os.path.exists(output_file):
    print(f"Video successfully saved at: {output_file}")
else:
    print("Error: Video file not saved")
