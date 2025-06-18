import cv2
import numpy as np
import imutils
import regex as re
import matplotlib.pyplot as plt
import os
import re

# Open video file
video_path='wp1.mp4'
video = video_path
cap = cv2.VideoCapture(video)
cnt = 0

# Check if video opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read the first frame
ret, first_frame = cap.read()

# Loop through video frames
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        roi = frame
        cv2.imshow("image", roi)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        
        # Save frames to the 'Frame' directory
        cv2.imwrite('Frame/' + str(cnt) + '.png', roi)
        cnt = cnt + 1
    else:
        break

# Close video stream
cv2.destroyAllWindows()

# Read frames from directory and sort by frame number
frames = os.listdir('Frame/')
frames.sort(key=lambda f: int(re.sub(r'\D', '', f)))

images = []
trajectory_points = []
radius_points = []
width_height = []

print(f"Processing {len(frames)} frames for ball detection...")

# Loop through frames
for frame_idx, frame_filename in enumerate(frames):
    frame_path = os.path.join('Frame/', frame_filename)
    img = cv2.imread(frame_path)
    result = img.copy()

    # Convert image to HSV
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for masking (targeting ball colors - adjust these for your ball)
    lower1 = np.array([0, 100, 0])      # Lower red range
    upper1 = np.array([10, 255, 255])   # Upper red range
    lower2 = np.array([160, 100, 20])   # Lower red range (wraparound)
    upper2 = np.array([180, 255, 255])  # Upper red range (wraparound)

    # Create masks and apply them
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)
    full_mask = lower_mask + upper_mask
    result = cv2.bitwise_and(result, result, mask=full_mask)

    # Find contours in the masked image
    contours, _ = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detectedCircles = []
    image = img.copy()

    # Loop through contours and detect circles
    for i, c in enumerate(contours):
        blobArea = cv2.contourArea(c)
        blobPerimeter = cv2.arcLength(c, True)

        if blobPerimeter != 0:
            blobCircularity = (4 * 3.1416 * blobArea) / (blobPerimeter**2)
            minCircularity = 0.2
            minArea = 35

            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)

            # Check circularity and area conditions
            if blobCircularity > minCircularity and blobArea > minArea:
                detectedCircles.append([center, radius, blobArea, blobCircularity])

    # Process detected circles
    if len(detectedCircles) != 0:
        largest_blob = max(detectedCircles, key=lambda x: x[2] * x[3])
        largest_center, largest_radius, largest_area, largest_circularity = largest_blob

        # Adjust for low circularity
        if largest_circularity < 0.4 and len(detectedCircles) > 1:
            remaining_blobs = [blob for blob in detectedCircles if blob != largest_blob]
            largest_circularity_blob = max(remaining_blobs, key=lambda x: x[2] * x[3])
            largest_center, largest_radius, largest_area, largest_circularity = largest_circularity_blob

        w, h = 4 * largest_radius, 4 * largest_radius
        color = (255, 0, 0)

        x, y = largest_center
        cv2.circle(image, largest_center, largest_radius, color, 2)
        cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 3)
        
        # Add detection info
        cv2.putText(image, f'Ball Found - Area: {largest_area:.0f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f'Circularity: {largest_circularity:.2f}', 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f'Radius: {largest_radius}px', 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        trajectory_points.append(largest_center)
        radius_points.append(largest_radius)
        width_height.append([w, h])
    else:
        # No ball detected in this frame
        cv2.putText(image, f'No ball detected (Frame {frame_idx+1})', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw trajectory if we have points
    if len(trajectory_points) > 1:
        for i in range(1, len(trajectory_points)):
            cv2.line(image, trajectory_points[i-1], trajectory_points[i], (0, 255, 255), 2)
    
    images.append(image)
    
    # Progress indicator
    if (frame_idx + 1) % 50 == 0:
        print(f"Processed {frame_idx + 1}/{len(frames)} frames...")

print(f"Ball detection completed! Found ball in {len(trajectory_points)}/{len(frames)} frames")

# Save processed images to 'Frame_b' directory
output_directory = 'Frame_b/'
os.makedirs(output_directory, exist_ok=True)
for i in range(len(images)):
    processed_image = images[i]
    cv2.imwrite(os.path.join(output_directory, f'processed_frame_{i:04d}.png'), processed_image)

print("Processed frames saved to Frame_b/ directory")

# Create slow motion video from processed frames
frames_output = os.listdir('Frame_b/')
frames_output.sort(key=lambda f: int(re.sub(r'\D', '', f)))
frame_array = []

for frame_file in frames_output:
    img = cv2.imread(os.path.join('Frame_b/', frame_file))
    if img is not None:
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

if frame_array:
    # Create slow motion video (adjust FPS for desired slow motion effect)
    slow_motion_fps = 8  # Lower = slower motion (original was probably 20-30 FPS)
    # slow_motion_fps = 5   # Very slow (6x slower)
    # slow_motion_fps = 12  # Moderately slow (2x slower)
    # slow_motion_fps = 15  # Slightly slow (1.5x slower)
    
    print(f"Creating slow motion video with {slow_motion_fps} FPS...")
    out = cv2.VideoWriter('wp1out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), slow_motion_fps, size)
    
    for frame in frame_array:
        out.write(frame)
    out.release()
    
    print(f"Slow motion video created: fv2out1_slowmo.mp4")
else:
    print("No frames found for video creation!")

# Print detection statistics
if trajectory_points:
    print(f"\n--- Ball Tracking Statistics ---")
    print(f"Total frames processed: {len(frames)}")
    print(f"Frames with ball detected: {len(trajectory_points)}")
    print(f"Detection rate: {len(trajectory_points)/len(frames)*100:.1f}%")
    print(f"Average ball radius: {np.mean(radius_points):.1f} pixels")
    print(f"Ball radius range: {min(radius_points)} - {max(radius_points)} pixels")
else:
    print("No ball detected in any frame!")

# Cleanup function to delete frame files and save disk space
def cleanup_directories():
    """Delete all frames from both directories after video creation"""
    directories_to_clean = ['Frame/', 'Frame_b/']
    total_deleted = 0
    total_size_saved = 0
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            try:
                files = os.listdir(directory)
                directory_size = 0
                
                for file in files:
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        # Get file size before deleting
                        file_size = os.path.getsize(file_path)
                        directory_size += file_size
                        os.remove(file_path)
                        total_deleted += 1
                
                total_size_saved += directory_size
                print(f"✓ Cleaned up {len(files)} files from {directory} (saved {directory_size/1024/1024:.1f} MB)")
                
            except Exception as e:
                print(f"✗ Error cleaning {directory}: {e}")
        else:
            print(f"Directory {directory} not found")
    
    print(f"\n--- Cleanup Summary ---")
    print(f"Total files deleted: {total_deleted}")
    print(f"Total disk space saved: {total_size_saved/1024/1024:.1f} MB")
    print("✓ Cleanup completed! Only the output video remains.")

# Ask user for cleanup confirmation (safety feature)
print(f"\n{'='*50}")
print("DISK SPACE CLEANUP")
print(f"{'='*50}")
print("The frame extraction created many image files that take up disk space.")
print("You can safely delete them now that the video has been created.")
print("\nFrame directories:")
print("- Frame/ (original extracted frames)")  
print("- Frame_b/ (processed frames with ball tracking)")

cleanup_choice = input("\nDo you want to delete all frame files to save disk space? (y/n): ").lower().strip()

if cleanup_choice in ['y', 'yes']:
    print("\nStarting cleanup...")
    cleanup_directories()
else:
    print("\nFrame files kept. You can manually delete the Frame/ and Frame_b/ folders later if needed.")
    print("Note: These folders may contain hundreds of images taking up significant disk space.")

print(f"\n{'='*50}")
print("PROCESSING COMPLETE!")
print(f"{'='*50}")
print(f"✓ Slow motion video saved as: fv2out1_slowmo.mp4")
print(f"✓ Video FPS: {slow_motion_fps} (slower than original)")
print(f"✓ Ball detection rate: {len(trajectory_points)/len(frames)*100:.1f}%")

# Color adjustment tips
print(f"\n--- Tips for Better Detection ---")
print("If ball detection is poor, try adjusting these HSV color ranges:")
print("For different colored balls:")
print("- Orange ball: lower1=[5,100,100], upper1=[15,255,255]")
print("- Yellow ball: lower1=[20,100,100], upper1=[30,255,255]") 
print("- Green ball: lower1=[40,100,100], upper1=[80,255,255]")
print("- Blue ball: lower1=[100,100,100], upper1=[130,255,255]")
print("- White ball: lower1=[0,0,200], upper1=[180,30,255]")