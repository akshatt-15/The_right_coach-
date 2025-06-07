import cv2
import numpy as np
import imutils
import regex as re
import matplotlib.pyplot as plt
import os
import math
from collections import deque
from sklearn.cluster import DBSCAN

class CombinedBallTracker:
    def __init__(self):
        # Tracking parameters
        self.previous_positions = deque(maxlen=10)
        self.previous_velocities = deque(maxlen=5)
        self.ball_size_history = deque(maxlen=10)
        
        # Detection parameters - will be auto-adjusted
        self.expected_ball_size_range = (8, 80)
        self.max_position_jump = 200
        self.min_detection_confidence = 0.3
        
        # Motion characteristics
        self.gravity_direction = None
        self.typical_ball_speed = None
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Video orientation
        self.video_orientation = 0  # Will be detected: 0, 90, 180, 270
        
        # Ball color ranges (HSV)
        self.color_ranges = {
            'red': [([0, 100, 50], [10, 255, 255]), ([160, 100, 50], [180, 255, 255])],
            'orange': [([10, 100, 100], [25, 255, 255])],
            'yellow': [([20, 100, 100], [40, 255, 255])],
            'green': [([40, 100, 100], [80, 255, 255])],
            'blue': [([100, 100, 100], [130, 255, 255])],
            'white': [([0, 0, 200], [180, 30, 255])]
        }
        
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
    
    def color_based_detection(self, frame):
        """Enhanced color-based detection with multiple ball colors"""
        candidates = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Apply all color ranges
        for color_name, ranges in self.color_ranges.items():
            for lower, upper in ranges:
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
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = (4 * np.pi * area) / (perimeter**2)
            
            # Filter by circularity (balls should be reasonably circular)
            if circularity < 0.2:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            
            # Filter out human body parts
            if self.is_human_body_part(contour, center, radius):
                continue
            
            candidates.append({
                'center': center,
                'radius': int(radius),
                'contour': contour,
                'area': area,
                'circularity': circularity,
                'method': 'color'
            })
        
        return candidates
    
    def motion_based_detection(self, frame):
        """Motion-based detection for moving balls"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in motion mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Reasonable size for moving ball
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                candidates.append({
                    'center': center,
                    'radius': int(radius),
                    'contour': contour,
                    'area': area,
                    'method': 'motion'
                })
        
        return candidates
    
    def hough_circle_detection(self, frame):
        """Hough circle detection for ball shapes"""
        candidates = []
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
    
    def is_trajectory_consistent(self, center):
        """Check trajectory consistency"""
        if len(self.previous_positions) < 2:
            return True
        
        # Check distance from last position
        last_pos = self.previous_positions[-1]
        distance = math.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
        
        if distance > self.max_position_jump:
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
    
    def calculate_confidence(self, candidate):
        """Calculate detection confidence score"""
        confidence = 0.0
        
        # Base confidence from detection method
        if candidate.get('method') == 'color':
            confidence += 0.4
            # Bonus for circularity
            if 'circularity' in candidate:
                confidence += candidate['circularity'] * 0.3
        elif candidate.get('method') == 'hough':
            confidence += 0.5
        elif candidate.get('method') == 'motion':
            confidence += 0.3
        
        # Trajectory consistency bonus
        if self.is_trajectory_consistent(candidate['center']):
            confidence += 0.2
        
        # Size consistency bonus
        if self.is_size_consistent(candidate['radius']):
            confidence += 0.2
        
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
    
    def detect_ball(self, frame, frame_idx):
        """Main ball detection function combining all methods"""
        # Rotate frame if needed
        frame = self.rotate_frame(frame, self.video_orientation)
        
        # Get candidates from all detection methods
        color_candidates = self.color_based_detection(frame)
        motion_candidates = self.motion_based_detection(frame)
        hough_candidates = self.hough_circle_detection(frame)
        
        # Combine all candidates
        all_candidates = color_candidates + motion_candidates + hough_candidates
        
        # Filter and validate candidates
        validated_candidates = []
        
        for candidate in all_candidates:
            center = candidate['center']
            radius = candidate['radius']
            
            # Skip if too close to frame edges
            h, w = frame.shape[:2]
            margin = 20
            if (center[0] < margin or center[0] > w-margin or 
                center[1] < margin or center[1] > h-margin):
                continue
            
            # Physics validation
            if not self.physics_based_validation(center, frame_idx):
                continue
            
            # Calculate confidence score
            confidence = self.calculate_confidence(candidate)
            candidate['confidence'] = confidence
            
            if confidence > self.min_detection_confidence:
                validated_candidates.append(candidate)
        
        # Select best candidate
        if validated_candidates:
            best_candidate = max(validated_candidates, key=lambda x: x['confidence'])
            self.update_tracking_history(best_candidate)
            return best_candidate
        
        return None
    
    def predict_ball_position(self):
        """Predict ball position based on trajectory"""
        if len(self.previous_positions) < 2:
            return None
        
        last_pos = self.previous_positions[-1]
        second_last = self.previous_positions[-2]
        
        # Calculate velocity
        dx = last_pos[0] - second_last[0]
        dy = last_pos[1] - second_last[1]
        
        # Predict next position
        predicted_x = last_pos[0] + dx
        predicted_y = last_pos[1] + dy
        
        # Apply gravity if detected
        if self.gravity_direction == 'down':
            predicted_y += 2
        elif self.gravity_direction == 'up':
            predicted_y -= 2
        
        avg_radius = int(np.mean(list(self.ball_size_history)) if self.ball_size_history else 15)
        
        return {
            'center': (predicted_x, predicted_y),
            'radius': avg_radius,
            'confidence': 0.2,
            'method': 'predicted'
        }

def process_video(video_path, output_name='output', slow_motion_fps=8, cleanup_frames=True):
    """Main function to process video with ball tracking"""
    
    print(f"Processing video: {video_path}")
    
    # Initialize tracker
    tracker = CombinedBallTracker()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
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
        tracker.detect_video_orientation(sample_frames)
    
    # Extract frames
    print("Extracting frames...")
    os.makedirs('Frame', exist_ok=True)
    cnt = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'Frame/{cnt}.png', frame)
            cnt += 1
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Process frames
    frames = os.listdir('Frame/')
    frames.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    
    images = []
    trajectory_points = []
    radius_points = []
    detection_confidence = []
    detection_methods = []
    
    print(f"Processing {len(frames)} frames with combined ball detection...")
    
    for i, frame_filename in enumerate(frames):
        frame_path = os.path.join('Frame/', frame_filename)
        img = cv2.imread(frame_path)
        
        if img is None:
            continue
        
        # Detect ball
        detection_result = tracker.detect_ball(img, i)
        
        if detection_result:
            center = detection_result['center']
            radius = detection_result['radius']
            confidence = detection_result['confidence']
            method = detection_result['method']
            
            # Draw detection
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 165, 255)
            cv2.circle(img, center, radius, color, 2)
            
            # Draw bounding rectangle
            w, h = 4 * radius, 4 * radius
            x, y = center
            cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 2)
            
            # Add detection info
            cv2.putText(img, f'Ball Found - {method.upper()}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Confidence: {confidence:.2f}', (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f'Radius: {radius}px', (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            trajectory_points.append(center)
            radius_points.append(radius)
            detection_confidence.append(confidence)
            detection_methods.append(method)
            
        else:
            # Try prediction
            predicted = tracker.predict_ball_position()
            if predicted:
                center = predicted['center']
                radius = predicted['radius']
                
                cv2.circle(img, center, radius, (128, 128, 128), 1)
                cv2.putText(img, "Predicted", (center[0] - 30, center[1] - radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                
                trajectory_points.append(center)
                radius_points.append(radius)
                detection_confidence.append(0.2)
                detection_methods.append('predicted')
            else:
                cv2.putText(img, f'No ball detected (Frame {i+1})', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw trajectory
        if len(trajectory_points) > 1:
            for j in range(1, min(len(trajectory_points), 15)):
                if j < len(trajectory_points):
                    cv2.line(img, trajectory_points[-j-1], trajectory_points[-j], (255, 0, 0), 2)
        
        images.append(img)
        
        if i % 50 == 0:
            print(f"Processed {i+1}/{len(frames)} frames")
    
    # Detect gravity direction from complete trajectory
    if len(trajectory_points) > 10:
        tracker.detect_gravity_direction(trajectory_points)
    
    # Save processed images
    output_directory = 'Frame_b/'
    os.makedirs(output_directory, exist_ok=True)
    for i, processed_image in enumerate(images):
        cv2.imwrite(os.path.join(output_directory, f'processed_frame_{i:04d}.png'), processed_image)
    
    # Create output video
    if images:
        height, width, layers = images[0].shape
        size = (width, height)
        
        output_video = f'{output_name}_tracked.mp4'
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), slow_motion_fps, size)
        for frame in images:
            out.write(frame)
        out.release()
        print(f"Output video saved as: {output_video}")
    
    # Print statistics
    print(f"\n{'='*50}")
    print("BALL TRACKING STATISTICS")
    print(f"{'='*50}")
    print(f"Total frames processed: {len(frames)}")
    print(f"Frames with ball detected: {len(trajectory_points)}")
    print(f"Detection rate: {len(trajectory_points)/len(frames)*100:.1f}%")
    
    if detection_confidence:
        avg_confidence = np.mean(detection_confidence)
        high_confidence_rate = len([c for c in detection_confidence if c > 0.6]) / len(detection_confidence) * 100
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"High confidence detections: {high_confidence_rate:.1f}%")
        
        # Method breakdown
        method_counts = {}
        for method in detection_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"\nDetection method breakdown:")
        for method, count in method_counts.items():
            percentage = count / len(detection_methods) * 100
            print(f"- {method.capitalize()}: {count} frames ({percentage:.1f}%)")
    
    if radius_points:
        print(f"Average ball radius: {np.mean(radius_points):.1f} pixels")
        print(f"Ball radius range: {min(radius_points)} - {max(radius_points)} pixels")
    
    print(f"Video orientation: {tracker.video_orientation}°")
    print(f"Gravity direction: {tracker.gravity_direction}")
    
    # Cleanup frames if requested
    if cleanup_frames:
        cleanup_directories()
    
    return output_video

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
                        file_size = os.path.getsize(file_path)
                        directory_size += file_size
                        os.remove(file_path)
                        total_deleted += 1
                
                total_size_saved += directory_size
                print(f"✓ Cleaned up {len(files)} files from {directory} (saved {directory_size/1024/1024:.1f} MB)")
                
            except Exception as e:
                print(f"✗ Error cleaning {directory}: {e}")
    
    print(f"\nTotal files deleted: {total_deleted}")
    print(f"Total disk space saved: {total_size_saved/1024/1024:.1f} MB")

# Main execution
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = 'Front view cut 1.mp4'  # Change this to your video file
    OUTPUT_NAME = 'combined_output'
    SLOW_MOTION_FPS = 8  # Lower = slower motion
    CLEANUP_FRAMES = True  # Set to False to keep frame files
    
    # Process the video
    try:
        output_video = process_video(
            video_path=VIDEO_PATH,
            output_name=OUTPUT_NAME,
            slow_motion_fps=SLOW_MOTION_FPS,
            cleanup_frames=CLEANUP_FRAMES
        )
        
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*50}")
        print(f"✓ Output video: {output_video}")
        print(f"✓ Video FPS: {SLOW_MOTION_FPS}")
        
        # Tips for improvement
        print(f"\n--- Tips for Better Detection ---")
        print("If ball detection is poor, you can:")
        print("1. Adjust color ranges in the CombinedBallTracker class")
        print("2. Modify detection confidence threshold (min_detection_confidence)")
        print("3. Adjust expected ball size range")
        print("4. Fine-tune motion detection parameters")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        print("Please check that the video file exists and is accessible.")
