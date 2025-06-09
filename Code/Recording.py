import cv2
import os
import time
from datetime import datetime

# Output path and file
path = '...camera' #put output path
output_file = os.path.join(path, 'filename.avi')
frame_per_minute = 1

os.makedirs(path, exist_ok=True)

def initialize_camera(width, height):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 128)  
    cam.set(cv2.CAP_PROP_CONTRAST, 128)   
    
    # Allow time for camera to initialize and focus
    time.sleep(3)
    return cam

initial_cam = initialize_camera(0, 0)  # Start with default resolution
ret, frame = initial_cam.read()
if not ret:
    print("Error: Could not read initial frame")
    initial_cam.release()
    exit()
frame_height, frame_width = frame.shape[:2]
initial_cam.release()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_file, fourcc, frame_per_minute, (frame_width, frame_height))

if not out.isOpened():
    print("Error: VideoWriter not opened!")
    exit()

print(f"Recording started. Saving 1 frame per minute to: {output_file}")
print(f"Frame size: {frame_width}x{frame_height}")

max_minutes = 360  # Run for up to 6 hours to protect the camera
frame_count = 0

# Also save individual frames for backup and analysis
frames_dir = os.path.join(path, 'frames')
os.makedirs(frames_dir, exist_ok=True)

try:
    while frame_count < max_minutes:
        # Completely reinitialize camera each time
        cam = initialize_camera(frame_width, frame_height)
        
        success = False
        for attempt in range(5):  # Retry up to 5 times
            ret, frame = cam.read()
            if ret:
                # Take a second frame after autofocus has had time to work
                time.sleep(0.5)
                ret, frame = cam.read()
                if ret:
                    success = True
                    break
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retry {attempt+1}: Failed to read frame")
            time.sleep(2)

        if success:
            # Save the frame to video
            out.write(frame)
            
            # Also save individual frame as image (backup)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            frame_file = os.path.join(frames_dir, f'frame_{frame_count:04d}_{timestamp}.jpg')
            cv2.imwrite(frame_file, frame)
            
            frame_count += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Frame {frame_count} captured. Waiting 60 seconds...")

            # Show live preview
            cv2.imshow('Live Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user (pressed 'q').")
                break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: Could not read frame after retries. Skipping frame.")

        # Release camera resources completely
        cam.release()
        time.sleep(60)

except KeyboardInterrupt:
    print("Recording interrupted by user.")

# Cleanup
out.release()
cv2.destroyAllWindows()

if os.path.exists(output_file):
    print(f"Video successfully saved at: {output_file}")
    print(f"Individual frames saved in: {frames_dir}")
else:
    print("Error: Video file not saved")
