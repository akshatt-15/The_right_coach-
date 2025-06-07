import cv2
import numpy as np
import imutils
import regex as re
import matplotlib.pyplot as plt
import os
import re

# Open video file
video_path='Front view cut 1.mp4'
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

# Loop through frames
for frame_filename in frames:
    frame_path = os.path.join('Frame/', frame_filename)
    img = cv2.imread(frame_path)
    result = img.copy()

    # Convert image to HSV
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for masking
    lower1 = np.array([0, 100, 0])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([180, 255, 255])

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
        trajectory_points.append(largest_center)
        radius_points.append(largest_radius)
        width_height.append([w, h])
        images.append(image)

# Save processed images to 'Frame_b' directory
output_directory = 'Frame_b/'
os.makedirs(output_directory, exist_ok=True)
for i in range(len(images)):
    processed_image = images[i]
    cv2.imwrite(os.path.join(output_directory, f'processed_frame_{i}.png'), processed_image)

# Create video from processed frames
frames = os.listdir('Frame_b/')
frames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
frame_array = []

for i in range(len(frames)):
    # Reading each file
    img = cv2.imread('Frame_b/' + frames[i])
    height, width, layers = img.shape
    size = (width, height)
    # Inserting the frames into an image array
    frame_array.append(img)

# Write video file
out = cv2.VideoWriter('frontout.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20 ,size)
for i in range(len(frame_array)):
    # Writing to an image array
    out.write(frame_array[i])
out.release() 
