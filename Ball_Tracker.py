import cv2
import numpy as np
import imutils
import regex as re
import matplotlib.pyplot as plt
import os
import math
from collections import deque
from sklearn.cluster import DBSCAN

class SmartBallDetector:
    def __init__(self):
        # Tracking parameters
        self.previous_positions = deque(maxlen=10)
        self.previous_velocities = deque(maxlen=5)
        self.ball_size_history = deque(maxlen=10)
        
        # Detection parameters - will be auto-adjusted
        self.expected_ball_size_range = (8, 80)
        self.max_position_jump = 200
        self.min_detection_confidence = 0.4
        
        # Motion characteristics
        self.gravity_direction = None  # Will be detected automatically
        self.typical_ball_speed = None
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Video orientation
        self.video_orientation = 0  # Will be detected: 0, 90, 180, 270
        
    def detect_video_orientation(self, frames_sample):
        """Detect if video is rotated and determine orientation"""
        orientations_score = {0: 0, 90: 0, 180: 0, 270: 0}
        
        for frame in frames_sample[:5]:  # Test first 5 frames
            for angle in [0, 90, 180, 270]:
                rotated = self.rotate_frame(frame, angle)
                h, w = rotated.shape[:2]
                
                # Check if this orientation makes sense (landscape vs portrait)
                if w > h:  # Landscape - typical for sports videos
                    orientations_score[angle] += 2
                
                # Detect motion direction to infer gravity
                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Count vertical vs horizontal edges
                vertical_edges = np.sum(edges[:, :w//4]) + np.sum(edges[:, 3*w//4:])
                horizontal_edges = np.sum(edges[:h//4, :]) + np.sum(edges[3*h//4:, :])
                
                if vertical_edges > horizontal_edges:
                    orientations_score[angle] += 1
        
        self.video_orientation = max(orientations_score, key=orientations_score.get)
        print(f"Detected video orientation: {self.video_orientation}°")
        return self.video_orientation
    
    def rotate_frame(self, frame, angle):
        """Rotate frame by specified angle"""
        if angle == 0:
            return frame
        elif angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def detect_gravity_direction(self, trajectory_points):
        """Detect gravity direction from ball trajectory"""
        if len(trajectory_points) < 10:
            return None
            
        # Analyze vertical movement patterns
        y_positions = [p[1] for p in trajectory_points[-10:]]
        y_velocities = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        
        # Check for acceleration patterns typical of gravity
        if len(y_velocities) >= 3:
            accelerations = [y_velocities[i+1] - y_velocities[i] for i in range(len(y_velocities)-1)]
            avg_acceleration = np.mean(accelerations)
            
            if abs(avg_acceleration) > 0.5:  # Significant acceleration
                self.gravity_direction = 'down' if avg_acceleration > 0 else 'up'
                return self.gravity_direction
        
        return None
    
    def is_human_body_part(self, contour, center, radius):
        """Advanced detection to filter out human body parts"""
        # Get contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return True
            
        # Calculate shape descriptors
        circularity = (4 * np.pi * area) / (perimeter**2)
        
        # Get convex hull properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        extent = area / (w * h)
        aspect_ratio = max(w, h) / min(w, h)
        
        # Human body parts typically have:
        # 1. Lower solidity (not completely filled)
        # 2. Higher aspect ratio (elongated)
        # 3. Lower extent (don't fill bounding rectangle)
        # 4. Irregular shape (low circularity but high area)
        
        # Elbow/joint detection
        if (solidity < 0.7 and aspect_ratio > 1.8) or (extent < 0.6 and area > 500):
            return True
            
        # Large irregular shapes (body parts)
        if area > 1000 and circularity < 0.3:
            return True
            
        # Very elongated objects
        if aspect_ratio > 3.0:
            return True
            
        return False
    
    def detect_motion_based_candidates(self, frame):
        """Use motion detection to find moving objects"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in motion mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Reasonable size for moving ball
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                motion_candidates.append({
                    'center': center,
                    'radius': int(radius),
                    'contour': contour,
                    'area': area
                })
        
        return motion_candidates, fg_mask
    
    def physics_based_validation(self, center, timestamp):
        """Validate detection based on physics"""
        if len(self.previous_positions) < 3:
            return True
            
        # Calculate trajectory physics
        positions = list(self.previous_positions)[-3:] + [center]
        
        # Check for realistic ball physics
        velocities = []
        for i in range(len(positions)-1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            velocity = math.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        # Ball should have consistent or gradually changing velocity
        if len(velocities) >= 2:
            velocity_change = abs(velocities[-1] - velocities[-2])
            max_velocity_change = 50  # pixels per frame
            
            if velocity_change > max_velocity_change:
                return False
        
        # Check for unrealistic speed
        if velocities and max(velocities) > 100:  # Too fast
            return False
            
        return True
    
    def advanced_ball_detection(self, frame, frame_idx):
        """Advanced ball detection with multiple validation layers"""
        # Rotate frame if needed
        frame = self.rotate_frame(frame, self.video_orientation)
        
        original_frame = frame.copy()
        
        # Method 1: Motion-based detection
        motion_candidates, motion_mask = self.detect_motion_based_candidates(frame)
        
        # Method 2: Color-based detection (improved)
        color_candidates = self.color_based_detection(frame)
        
        # Method 3: Template matching for specific ball types
        template_candidates = self.template_matching_detection(frame)
        
        # Combine all candidates
        all_candidates = motion_candidates + color_candidates + template_candidates
        
        # Advanced filtering
        validated_candidates = []
        
        for candidate in all_candidates:
            center = candidate['center']
            radius = candidate['radius']
            contour = candidate.get('contour')
            
            # Skip if too close to frame edges (likely partial objects)
            h, w = frame.shape[:2]
            margin = 20
            if (center[0] < margin or center[0] > w-margin or 
                center[1] < margin or center[1] > h-margin):
                continue
            
            # Filter out human body parts
            if contour is not None and self.is_human_body_part(contour, center, radius):
                continue
            
            # Physics validation
            if not self.physics_based_validation(center, frame_idx):
                continue
            
            # Trajectory consistency
            if not self.is_trajectory_consistent(center):
                continue
            
            # Size consistency
            if not self.is_size_consistent(radius):
                continue
            
            # Calculate confidence score
            confidence = self.calculate_confidence(candidate, frame)
            candidate['confidence'] = confidence
            
            if confidence > self.min_detection_confidence:
                validated_candidates.append(candidate)
        
        # Select best candidate
        if validated_candidates:
            best_candidate = max(validated_candidates, key=lambda x: x['confidence'])
            
            # Update tracking history
            self.update_tracking_history(best_candidate)
            
            return best_candidate
        
        return None
    
    def color_based_detection(self, frame):
        """Improved color-based detection"""
        candidates = []
        
        # Convert to multiple color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Multiple color ranges for different ball types
        color_ranges = [
            # Red ball (cricket)
            ([0, 100, 50], [10, 255, 255]),
            ([160, 100, 50], [180, 255, 255]),
            # Orange ball
            ([10, 100, 100], [25, 255, 255]),
            # Yellow ball (tennis)
            ([20, 100, 100], [40, 255, 255]),
            # White ball (cricket)
            ([0, 0, 200], [180, 30, 255]),
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:
                continue
                
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            
            candidates.append({
                'center': center,
                'radius': int(radius),
                'contour': contour,
                'area': area,
                'method': 'color'
            })
        
        return candidates
    
    def template_matching_detection(self, frame):
        """Template matching for specific ball patterns"""
        # This is a placeholder - you can add specific ball templates
        candidates = []
        
        # Use HoughCircles as a form of template matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=25,
            minRadius=self.expected_ball_size_range[0],
            maxRadius=self.expected_ball_size_range[1]
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                candidates.append({
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r**2,
                    'method': 'hough',
                    'contour': None
                })
        
        return candidates
    
    def is_trajectory_consistent(self, center):
        """Check trajectory consistency with improved logic"""
        if len(self.previous_positions) < 2:
            return True
        
        # Check distance from last position
        last_pos = self.previous_positions[-1]
        distance = math.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
        
        if distance > self.max_position_jump:
            return False
        
        # Check for consistent motion direction
        if len(self.previous_positions) >= 3:
            positions = list(self.previous_positions)[-3:] + [center]
            
            # Calculate motion smoothness
            direction_changes = 0
            for i in range(len(positions)-2):
                v1 = (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
                v2 = (positions[i+2][0] - positions[i+1][0], positions[i+2][1] - positions[i+1][1])
                
                # Check if direction change is too abrupt
                if abs(v1[0]) > 0 and abs(v1[1]) > 0 and abs(v2[0]) > 0 and abs(v2[1]) > 0:
                    angle_diff = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
                    if abs(angle_diff) > math.pi/2:  # More than 90 degree change
                        direction_changes += 1
            
            if direction_changes > 1:  # Too many abrupt changes
                return False
        
        return True
    
    def is_size_consistent(self, radius):
        """Check size consistency"""
        if not self.ball_size_history:
            return True
        
        avg_size = np.mean(list(self.ball_size_history))
        size_diff = abs(radius - avg_size)
        
        # Allow 50% size variation
        if size_diff > avg_size * 0.5:
            return False
        
        return True
    
    def calculate_confidence(self, candidate, frame):
        """Calculate detection confidence score"""
        confidence = 0.0
        
        # Base confidence from detection method
        if candidate.get('method') == 'color':
            confidence += 0.3
        elif candidate.get('method') == 'hough':
            confidence += 0.4
        else:
            confidence += 0.2
        
        # Trajectory consistency bonus
        if self.is_trajectory_consistent(candidate['center']):
            confidence += 0.3
        
        # Size consistency bonus
        if self.is_size_consistent(candidate['radius']):
            confidence += 0.2
        
        # Circularity bonus (if contour available)
        if candidate.get('contour') is not None:
            contour = candidate['contour']
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter**2)
                confidence += circularity * 0.2
        
        return min(confidence, 1.0)
    
    def update_tracking_history(self, candidate):
        """Update tracking history"""
        self.previous_positions.append(candidate['center'])
        self.ball_size_history.append(candidate['radius'])
        
        # Update expected size range based on history
        if len(self.ball_size_history) >= 5:
            sizes = list(self.ball_size_history)
            avg_size = np.mean(sizes)
            std_size = np.std(sizes)
            self.expected_ball_size_range = (
                max(5, int(avg_size - 2*std_size)),
                min(100, int(avg_size + 2*std_size))
            )

# Main execution code
detector = SmartBallDetector()

# Open video file
video_path = 'Front view cut 1.mp4'
cap = cv2.VideoCapture(video_path)
cnt = 0

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Read sample frames for orientation detection
sample_frames = []
for i in range(10):
    ret, frame = cap.read()
    if ret:
        sample_frames.append(frame)
    else:
        break

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Detect video orientation
if sample_frames:
    detector.detect_video_orientation(sample_frames)

# Extract all frames first
print("Extracting frames...")
os.makedirs('Frame', exist_ok=True)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'Frame/{cnt}.png', frame)
        cnt += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Process frames with advanced detection
frames = os.listdir('Frame/')
frames.sort(key=lambda f: int(re.sub(r'\D', '', f)))

images = []
trajectory_points = []
radius_points = []
detection_confidence = []

print(f"Processing {len(frames)} frames with advanced detection...")

for i, frame_filename in enumerate(frames):
    frame_path = os.path.join('Frame/', frame_filename)
    img = cv2.imread(frame_path)
    
    if img is None:
        continue
    
    # Advanced detection
    detection_result = detector.advanced_ball_detection(img, i)
    
    if detection_result:
        center = detection_result['center']
        radius = detection_result['radius']
        confidence = detection_result['confidence']
        
        # Draw detection
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        cv2.circle(img, center, radius, color, 2)
        
        # Draw trajectory
        if len(trajectory_points) > 1:
            for j in range(1, min(len(trajectory_points), 10)):
                cv2.line(img, trajectory_points[-j-1], trajectory_points[-j], (255, 0, 0), 1)
        
        # Add confidence and method text
        method = detection_result.get('method', 'multi')
        cv2.putText(img, f"{method}: {confidence:.2f}", 
                   (center[0] - 50, center[1] - radius - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        trajectory_points.append(center)
        radius_points.append(radius)
        detection_confidence.append(confidence)
    else:
        # Prediction based on trajectory
        if len(trajectory_points) >= 2:
            last_pos = trajectory_points[-1]
            second_last = trajectory_points[-2]
            predicted_x = last_pos[0] + (last_pos[0] - second_last[0])
            predicted_y = last_pos[1] + (last_pos[1] - second_last[1])
            
            # Apply gravity if detected
            if detector.gravity_direction == 'down':
                predicted_y += 2  # Gravity acceleration
            elif detector.gravity_direction == 'up':
                predicted_y -= 2
            
            predicted_center = (predicted_x, predicted_y)
            avg_radius = int(np.mean(radius_points[-5:]) if radius_points else 15)
            
            cv2.circle(img, predicted_center, avg_radius, (128, 128, 128), 1)
            cv2.putText(img, "Predicted", (predicted_x - 30, predicted_y - avg_radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            trajectory_points.append(predicted_center)
            radius_points.append(avg_radius)
            detection_confidence.append(0.2)
    
    images.append(img)
    
    if i % 20 == 0:
        print(f"Processed {i+1}/{len(frames)} frames")

# Detect gravity direction from complete trajectory
if len(trajectory_points) > 10:
    detector.detect_gravity_direction(trajectory_points)

# Save processed images
output_directory = 'Frame_b/'
os.makedirs(output_directory, exist_ok=True)
for i, processed_image in enumerate(images):
    cv2.imwrite(os.path.join(output_directory, f'processed_frame_{i}.png'), processed_image)

# Create output video
if images:
    height, width, layers = images[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter('frontout.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    for frame in images:
        out.write(frame)
    out.release()
    print("Output video saved as 'frontout.mp4'")

# Print statistics
if detection_confidence:
    avg_confidence = np.mean(detection_confidence)
    high_confidence_rate = len([c for c in detection_confidence if c > 0.6]) / len(detection_confidence) * 100
    print(f"\nDetection Statistics:")
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"High confidence detections: {high_confidence_rate:.1f}%")
    print(f"Video orientation: {detector.video_orientation}°")
    print(f"Gravity direction: {detector.gravity_direction}")
