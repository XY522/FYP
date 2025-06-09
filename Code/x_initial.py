import cv2
import numpy as np
import os

# Input video file path
input_path = "..."

# Check if file exists
if not os.path.exists(input_path):
    print("File not found:", input_path)
    exit()

# Try to open video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Dummy calibration data (replace with real if available)
camera_matrix = np.array([[800, 0, 640],
                          [0, 800, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0], dtype=np.float32)

# Get video info
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare undistortion maps
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs,
                                                     (frame_width, frame_height), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None,
                                         new_camera_matrix,
                                         (frame_width, frame_height), cv2.CV_16SC2)

# Read first frame and undistort
ret, first_frame = cap.read()
if not ret:
    print("Failed to read first frame.")
    cap.release()
    exit()

undistorted_first = cv2.remap(first_frame, map1, map2, interpolation=cv2.INTER_LINEAR)

# ROTATION CORRECTION STEP
print("Select two points to define the horizontal line (the line will be rotated to horizontal)")
rotation_img = undistorted_first.copy()
rotation_points = []

def get_rotation_line(event, x_coord, y_coord, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(rotation_points) < 2:
        rotation_points.append((x_coord, y_coord))
        cv2.circle(rotation_img, (x_coord, y_coord), 5, (0, 0, 255), -1)
        if len(rotation_points) == 2:
            cv2.line(rotation_img, rotation_points[0], rotation_points[1], (0, 255, 0), 2)
        cv2.imshow("Rotation Correction", rotation_img)

cv2.imshow("Rotation Correction", rotation_img)
cv2.setMouseCallback("Rotation Correction", get_rotation_line)

# Wait until two points are selected
while len(rotation_points) < 2:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        print("Rotation correction aborted. No rotation will be applied.")
        rotation_angle = 0
        break

# Calculate rotation angle if points were selected
if len(rotation_points) == 2:
    dx = rotation_points[1][0] - rotation_points[0][0]
    dy = rotation_points[1][1] - rotation_points[0][1]
    rotation_angle = np.degrees(np.arctan2(dy, dx))
    print(f"Detected rotation angle: {rotation_angle:.2f} degrees")
    
    apply_rotation = input(f"Apply {rotation_angle:.2f} degree rotation to correct? (y/n): ").lower() == 'y'
    if not apply_rotation:
        rotation_angle = 0
        print("Rotation correction skipped.")
else:
    rotation_angle = 0

cv2.destroyWindow("Rotation Correction")

# Apply rotation to the undistorted image
if rotation_angle != 0:
    center = (frame_width // 2, frame_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    undistorted_first = cv2.warpAffine(undistorted_first, rotation_matrix, (frame_width, frame_height), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print("Applied rotation correction")

# Calibration: Select two points for scale reference
print("Select two points on the ruler for distance calibration (Click two points)")
ruler_roi_img = undistorted_first.copy()
calibration_points = []

def get_calibration_point(event, x_coord, y_coord, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 2:
        calibration_points.append((x_coord, y_coord))
        cv2.circle(ruler_roi_img, (x_coord, y_coord), 5, (0, 0, 255), -1)
        if len(calibration_points) == 2:
            cv2.line(ruler_roi_img, calibration_points[0], calibration_points[1], (0, 255, 0), 2)
        cv2.imshow("Calibration", ruler_roi_img)

cv2.imshow("Calibration", ruler_roi_img)
cv2.setMouseCallback("Calibration", get_calibration_point)

# Wait until two points are selected
while len(calibration_points) < 2:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit if needed
        print("Calibration aborted. Using default scale.")
        calibration_points = [(0, 0), (100, 0)]
        break

# Calculate scale factor
if len(calibration_points) == 2:
    pixel_distance = np.sqrt((calibration_points[1][0] - calibration_points[0][0])**2 + 
                            (calibration_points[1][1] - calibration_points[0][1])**2)
    
    real_distance = float(input(f"Enter the real-world distance in mm between the two points (pixel distance: {pixel_distance:.2f}): "))
    scale_factor = real_distance / pixel_distance  # mm/pixel
    print(f"Calibration: {scale_factor:.6f} mm/pixel")
else:
    scale_factor = 0.1  # Default 0.1 mm/pixel
    print(f"Using default calibration: {scale_factor:.6f} mm/pixel")

cv2.destroyWindow("Calibration")

# Select ROI
print("Select ROI containing the reference point and gel")
roi = cv2.selectROI("Select Region of Interest", undistorted_first, fromCenter=False, showCrosshair=True)
x, y, w, h = roi
print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")
cv2.destroyWindow("Select Region of Interest")

# Crop to ROI
cropped = undistorted_first[y:y+h, x:x+w]

# === Select fixed reference point ===
print("Select the fixed reference point (marked in red in the image)")
ref_img = cropped.copy()
ref_point = None

def get_reference_point(event, x_coord, y_coord, flags, param):
    global ref_point
    if event == cv2.EVENT_LBUTTONDOWN and ref_point is None:
        ref_point = (x_coord, y_coord)
        cv2.circle(ref_img, ref_point, 5, (0, 0, 255), -1)
        cv2.imshow("Select Reference Point", ref_img)

cv2.imshow("Select Reference Point", ref_img)
cv2.setMouseCallback("Select Reference Point", get_reference_point)

# Wait until the reference point is selected
while ref_point is None:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        print("Reference point selection aborted. Using center of the top edge.")
        ref_point = (w // 2, 10)
        break

print(f"Selected reference point: x={ref_point[0]}, y={ref_point[1]}")
cv2.destroyWindow("Select Reference Point")

# Thresholding to find gel
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# Adaptive thresholding
binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

# Otsu's method
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Combine both methods
binary = cv2.bitwise_or(binary_adaptive, binary_otsu)

# Apply morphological operations
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Find intersection point and calculate distance ===
if contours:
    # Find the largest contour (gel)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Center line (vertical line from reference point)
    cx = ref_point[0]
    
    # Find topmost intersection
    contour_points = largest_contour[:, 0, :]
    cx_mask = np.abs(contour_points[:, 0] - cx) <= 5  # 5 pixel margin
    center_points = contour_points[cx_mask]
    
    if len(center_points) > 0:
        topmost_y = np.min(center_points[:, 1])
        
        # Calculate distance
        if ref_point[1] <= topmost_y:
            dist_pixel = topmost_y - ref_point[1]
        else:
            dist_pixel = 0
            
        dist_mm = dist_pixel * scale_factor
        
        # Create visualization
        result_frame = cropped.copy()
        cv2.drawContours(result_frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(result_frame, ref_point, 4, (255, 0, 0), -1)  # Blue dot for reference
        cv2.circle(result_frame, (cx, topmost_y), 4, (0, 0, 255), -1)  # Red dot for gel surface
        cv2.line(result_frame, ref_point, (cx, topmost_y), (0, 255, 255), 2)  # Yellow line
        
        # Display result
        cv2.imshow("Distance Measurement Result", result_frame)
        
        # Print results to terminal
        print("="*50)
        print("DISTANCE MEASUREMENT RESULTS")
        print("="*50)
        print(f"Reference point: ({ref_point[0]}, {ref_point[1]}) pixels")
        print(f"Gel surface point: ({cx}, {topmost_y}) pixels")
        print(f"Distance in pixels: {dist_pixel:.2f} px")
        print(f"Distance in real world: {dist_mm:.3f} mm")
        print(f"Scale factor used: {scale_factor:.6f} mm/pixel")
        print("="*50)
        
        print("Press any key to exit...")
        cv2.waitKey(0)
        
    else:
        print("Could not find intersection between reference line and gel surface")
else:
    print("No contours found in the thresholded image")

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

print("Distance measurement completed!")
