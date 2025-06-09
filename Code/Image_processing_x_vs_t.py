import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from datetime import timedelta

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
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            # Draw the line between the two points
            cv2.line(rotation_img, rotation_points[0], rotation_points[1], (0, 255, 0), 2)
        cv2.imshow("Rotation Correction", rotation_img)

cv2.imshow("Rotation Correction", rotation_img)
cv2.setMouseCallback("Rotation Correction", get_rotation_line)

# Wait until two points are selected
while len(rotation_points) < 2:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        print("Rotation correction aborted. No rotation will be applied.")
        rotation_points = [(0, 0), (100, 0)]  # Default horizontal line
        rotation_angle = 0
        break

# Calculate rotation angle if points were selected
if len(rotation_points) == 2:
    # Calculate angle between the line and the horizontal
    dx = rotation_points[1][0] - rotation_points[0][0]
    dy = rotation_points[1][1] - rotation_points[0][1]
    rotation_angle = np.degrees(np.arctan2(dy, dx))
    print(f"Detected rotation angle: {rotation_angle:.2f} degrees")
    
    # Flip the sign of the rotation - apply opposite direction
    correction_angle = rotation_angle  # This is what we'll apply (positive value rotates clockwise)
    
    # Confirm rotation
    apply_rotation = input(f"Apply {correction_angle:.2f} degree rotation to correct? (y/n): ").lower() == 'y'
    if not apply_rotation:
        rotation_angle = 0
        print("Rotation correction skipped.")
else:
    rotation_angle = 0

cv2.destroyWindow("Rotation Correction")

# Apply rotation to the undistorted image
if rotation_angle != 0:
    # Get rotation matrix
    center = (frame_width // 2, frame_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Apply rotation
    undistorted_first = cv2.warpAffine(undistorted_first, rotation_matrix, (frame_width, frame_height), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print("Applied rotation correction")

# Select ROI on the rotated frame
roi = cv2.selectROI("Select Region of Interest", undistorted_first, fromCenter=False, showCrosshair=True)
x, y, w, h = roi
print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")

# Calibration: Select two points for scale reference
print("Select two points on the ruler for distance calibration (Click two points)")
ruler_roi_img = undistorted_first.copy()
calibration_points = []

def get_calibration_point(event, x_coord, y_coord, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 2:
        calibration_points.append((x_coord, y_coord))
        cv2.circle(ruler_roi_img, (x_coord, y_coord), 5, (0, 0, 255), -1)
        if len(calibration_points) == 2:
            # Draw a line between the two calibration points
            cv2.line(ruler_roi_img, calibration_points[0], calibration_points[1], (0, 255, 0), 2)
        cv2.imshow("Calibration", ruler_roi_img)

cv2.imshow("Calibration", ruler_roi_img)
cv2.setMouseCallback("Calibration", get_calibration_point)

# Wait until two points are selected
while len(calibration_points) < 2:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit if needed
        print("Calibration aborted. Using default scale.")
        calibration_points = [(0, 0), (100, 0)]  # Default 100 pixels
        break

# Ask for real-world distance
if len(calibration_points) == 2:
    pixel_distance = np.sqrt((calibration_points[1][0] - calibration_points[0][0])**2 + 
                            (calibration_points[1][1] - calibration_points[0][1])**2)
    
    # You can modify this to use a GUI input dialog if preferred
    real_distance = float(input(f"Enter the real-world distance in mm between the two points (pixel distance: {pixel_distance:.2f}): "))
    scale_factor = real_distance / pixel_distance  # mm/pixel
    print(f"Calibration: {scale_factor:.6f} mm/pixel")
else:
    scale_factor = 0.1  # Default 0.1 mm/pixel
    print(f"Using default calibration: {scale_factor:.6f} mm/pixel")

cv2.destroyWindow("Calibration")

# Reset to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("Processing: Undistort → Rotate → Crop → Threshold → Contour → Reference Point Tracking")

# Data storage for tracking
frame_numbers = []
time_points = []  # in minutes (assuming 1 frame per minute)
y_positions_pixel = []
y_positions_mm = []
reference_y = None  # Reference y position for displacement calculation

# For tracking consistency
last_topmost_y = None
last_valid_frame = -1
max_frames_to_interpolate = 5  # Maximum number of frames to interpolate if tracking is lost

# Create rotation matrix for all frames
rotation_matrix = None
if rotation_angle != 0:
    center = (frame_width // 2, frame_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

# Debug mode flag
debug_mode = True  # Set to False to disable debug info

# Improved functions for contour detection and centerline intersection
def find_topmost_intersection(contour, cx, search_height, margin=5):
    """
    Find the topmost intersection between the contour and the centerline (±margin).
    
    Args:
        contour: The contour to analyze
        cx: x-coordinate of the centerline
        search_height: The height to search within
        margin: How many pixels to consider on each side of the centerline
        
    Returns:
        The y-coordinate of the topmost intersection or None if not found
    """
    # Try to find contour points near the centerline in the upper region
    contour_points = contour[:, 0, :]  # shape (n_points, 2)
    
    # Mask for points near the centerline
    cx_mask = np.abs(contour_points[:, 0] - cx) <= margin
    # Mask for points in the upper part of the image (lower y values)
    y_mask = contour_points[:, 1] < search_height
    
    # Combine masks to get points that meet both criteria
    combined_mask = np.logical_and(cx_mask, y_mask)
    center_points = contour_points[combined_mask]
    
    if len(center_points) > 0:
        # Find the topmost (smallest y) point
        topmost_idx = np.argmin(center_points[:, 1])
        return center_points[topmost_idx, 1]
    
    return None

def find_precise_intersection(contour, cx, search_height, h):
    """
    Try multiple methods to find a reliable intersection point.
    
    Args:
        contour: The contour to analyze
        cx: x-coordinate of the centerline
        search_height: The height to search within
        h: Image height
        
    Returns:
        The y-coordinate of the topmost intersection or None if not found
    """
    # Method 1: Try with a small margin first (more precise)
    topmost_y = find_topmost_intersection(contour, cx, search_height, margin=3)
    
    # Method 2: If not found, try with a larger margin
    if topmost_y is None:
        topmost_y = find_topmost_intersection(contour, cx, search_height, margin=7)
    
    # Method 3: If still not found, try to find any point in the upper third
    if topmost_y is None:
        # Get bounding rectangle and check if it crosses the centerline
        x, y, w, h_rect = cv2.boundingRect(contour)
        if x <= cx <= x + w and y < search_height:
            # Create a vertical line and find intersection with contour
            line_img = np.zeros((h, 1), dtype=np.uint8)
            contour_img = np.zeros((h, 1), dtype=np.uint8)
            cv2.drawContours(contour_img, [contour], -1, 255, 1, offset=(-cx, 0))
            cv2.line(line_img, (0, 0), (0, h), 255, 1)
            intersection = np.logical_and(line_img, contour_img)
            intersection_pts = np.where(intersection)[0]
            if len(intersection_pts) > 0:
                topmost_y = np.min(intersection_pts)
    
    return topmost_y

# Frame-by-frame processing
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing all frames.")
        break
    
    # Undistort frame
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    
    # Apply rotation if needed
    if rotation_matrix is not None:
        undistorted = cv2.warpAffine(undistorted, rotation_matrix, (frame_width, frame_height), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if debug_mode and frame_count == 0:
            cv2.imshow("After Rotation", undistorted)
            cv2.waitKey(1000)  # Show the rotated image for 1 second on first frame
    
    # Crop to ROI
    cropped = undistorted[y:y+h, x:x+w]
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Try adaptive thresholding instead of Otsu's method
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # Also try Otsu's method
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine both methods for better results
    binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    # For debugging: show the thresholded image on the first frame
    if debug_mode and frame_count == 0:
        cv2.imshow("Thresholded Image", binary)
        cv2.waitKey(1000)  # Show for 1 second
    
    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a clean mask with only the gel
    clean_mask = np.zeros_like(binary)
    
    # Filter contours based on area to keep only the gel
    min_contour_area = 300  # Reduced minimum area to accommodate rotation effects
    
    # First, find the largest contour (assumed to be the gel)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # Only keep contours that are at least 5% of the largest one (more lenient)
        area_threshold = max(min_contour_area, largest_area * 0.05)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > area_threshold:
                cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
    
    # Apply the cleaned mask to the binary image
    binary_cleaned = np.zeros_like(binary)
    binary_cleaned[clean_mask > 0] = 255
    
    # Use the cleaned binary image for further contour detection
    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    topmost_y = None
    
    if contours:
        # Find the largest contour (gel)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Center line (vertical line through image center)
        cx = w // 2
        
        # Use improved method to find topmost intersection
        search_height = h // 2  # Increased from h/3 to h/2 for better detection
        topmost_y = find_precise_intersection(largest_contour, cx, search_height, h)
        
        # If still not found and we have previous tracking data, try interpolation
        if topmost_y is None and last_topmost_y is not None:
            if frame_count - last_valid_frame <= max_frames_to_interpolate:
                # Use the last valid position as an approximation
                topmost_y = last_topmost_y
                print(f"Frame {frame_count}: Using previous position {topmost_y}")
        
        if topmost_y is not None:
            # Store data
            frame_numbers.append(frame_count)
            time_points.append(frame_count)  # 1 frame per minute, so frame count = minutes
            y_positions_pixel.append(topmost_y)
            
            # Calculate mm position (using scale_factor) 
            y_mm = topmost_y * scale_factor
            y_positions_mm.append(y_mm)
            
            # Set reference point if first detection
            if reference_y is None and topmost_y is not None:
                reference_y = y_mm
            
            # Update tracking variables
            last_topmost_y = topmost_y
            last_valid_frame = frame_count
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break
        
    frame_count += 1

# Cleanup video resources
cap.release()
cv2.destroyAllWindows()

# Calculate displacements relative to the initial position
# Convert to numpy array for easier processing
y_mm_array = np.array(y_positions_mm)
if reference_y is not None:
    # We need to take the absolute value to ensure positive values
    # Downward movement (y_mm_array > reference_y) should be positive
    displacements_mm = np.abs(y_mm_array - reference_y)  # Absolute value ensures positive
else:
    displacements_mm = np.zeros_like(y_mm_array)
    print("Warning: No reference point was detected")

# Save data to CSV
input_folder = os.path.dirname(input_path)
csv_path = os.path.join(input_folder, "displacement_data.csv")
with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame', 'Time (min)', 'Y Position (pixels)', 'Y Position (mm)', 'Displacement (mm)'])
    for i in range(len(frame_numbers)):
        csv_writer.writerow([
            frame_numbers[i],
            time_points[i],
            y_positions_pixel[i],
            y_positions_mm[i],
            displacements_mm[i]
        ])

print(f"Data saved to: {csv_path}")

# Create displacement vs time plot with positive direction downwards
plt.figure(figsize=(10, 6))
plt.plot(time_points, displacements_mm, 'b-', linewidth=2, marker='o', markersize=4)

# Set y-axis to start from 0 to emphasize positive values
plt.ylim(bottom=0)  # Force y-axis to start at 0

# Keep a clean plot but add just the trend value
plt.title('Displacement vs Time (Positive = Downward)')  # Add title to clarify direction
plt.xlabel('Time (min)')  # Add x-axis label for clarity
plt.ylabel('Displacement (mm)')  # Add y-axis label for clarity
plt.grid(True, linestyle='--', alpha=0.7)

# Add trendline with label showing just the trend value
if len(time_points) > 2:
    z = np.polyfit(time_points, displacements_mm, 1)
    p = np.poly1d(z)
    plt.plot(time_points, p(time_points), "r--", linewidth=1, 
             label=f"Trend: {z[0]:.4f} mm/min")
    plt.legend()

# Save plot with trend value
plot_path = os.path.join(input_folder, "displacement_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print("Process completed successfully!")
print(f"Plot saved to: {plot_path}")
print(f"Data saved to: {csv_path}")
