import cv2
import os
import sys
import logging
from datetime import datetime, timedelta
import argparse

class CompleteVideoClipper:
    def __init__(self, log_level=logging.INFO):
        """Initialize CompleteVideoClipper with logging configuration."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self, log_level):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'logs/video_clipper_{timestamp}.log'
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def get_video_info(self, video_path):
        """Get comprehensive video information."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            video_info = {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration_seconds': duration_seconds,
                'duration_formatted': str(timedelta(seconds=int(duration_seconds)))
            }
            
            cap.release()
            self.logger.info(f"Video info retrieved for {video_path}")
            return video_info
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            return None

    def frames_to_time(self, frame_number, fps):
        """Convert frame number to time format."""
        if fps <= 0:
            return "00:00:00"
        seconds = frame_number / fps
        return str(timedelta(seconds=int(seconds)))

    def time_to_frames(self, time_str, fps):
        """Convert time string (HH:MM:SS or MM:SS or SS) to frame number."""
        try:
            parts = time_str.split(':')
            if len(parts) == 1:  # SS
                total_seconds = int(parts[0])
            elif len(parts) == 2:  # MM:SS
                total_seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return None
            
            return int(total_seconds * fps)
        except:
            return None

    def display_video_info(self, video_info, video_path):
        """Display formatted video information."""
        print("=" * 80)
        print(f"VIDEO ANALYSIS: {os.path.basename(video_path)}")
        print("=" * 80)
        print(f"📁 File Path: {video_path}")
        print(f"📊 Total Frames: {video_info['total_frames']:,}")
        print(f"🎬 Frame Rate (FPS): {video_info['fps']:.2f}")
        print(f"📐 Resolution: {video_info['width']}x{video_info['height']}")
        print(f"⏱️  Total Duration: {video_info['duration_formatted']} ({video_info['duration_seconds']:.2f} seconds)")
        print("=" * 80)
        
        # Show some helpful frame reference points
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        print("\n🕐 FRAME REFERENCE GUIDE:")
        print("-" * 50)
        reference_points = [
            (0, "Start"),
            (int(fps * 30), "30 seconds"),
            (int(fps * 60), "1 minute"),
            (int(fps * 120), "2 minutes"),
            (int(fps * 300), "5 minutes"),
            (int(fps * 600), "10 minutes"),
            (total_frames // 4, "25% through"),
            (total_frames // 2, "50% through (middle)"),
            (total_frames * 3 // 4, "75% through"),
            (total_frames - 1, "End")
        ]
        
        for frame, description in reference_points:
            if frame <= total_frames:
                time_str = self.frames_to_time(frame, fps)
                print(f"Frame {frame:6,} = {time_str} ({description})")
        
        print("-" * 50)

    def extract_clip(self, input_path, output_path, start_frame, duration_seconds):
        """Extract a clip from the video starting at specified frame for given duration."""
        try:
            self.logger.info(f"Starting clip extraction: {output_path}")
            self.logger.info(f"Start frame: {start_frame}, Duration: {duration_seconds}s")
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open input video: {input_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate end frame
            frames_to_extract = int(duration_seconds * fps)
            end_frame = start_frame + frames_to_extract
            
            # Validate frame range
            if start_frame >= total_frames:
                self.logger.error(f"Start frame {start_frame} exceeds total frames {total_frames}")
                cap.release()
                return False
            
            if end_frame > total_frames:
                end_frame = total_frames
                actual_duration = (end_frame - start_frame) / fps
                self.logger.warning(f"Adjusted end frame to {end_frame}, actual duration: {actual_duration:.2f}s")
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.logger.error(f"Could not create output video: {output_path}")
                cap.release()
                return False
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract frames
            current_frame = start_frame
            frames_written = 0
            
            print(f"  Extracting frames {start_frame} to {end_frame}...")
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Could not read frame {current_frame}")
                    break
                
                out.write(frame)
                frames_written += 1
                current_frame += 1
                
                # Show progress every 100 frames
                if frames_written % 100 == 0:
                    progress = (frames_written / frames_to_extract) * 100
                    print(f"    Progress: {progress:.1f}% ({frames_written}/{frames_to_extract} frames)")
            
            # Cleanup
            cap.release()
            out.release()
            
            self.logger.info(f"Clip extraction completed: {frames_written} frames written")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during clip extraction: {str(e)}")
            return False

    def parse_clips_input(self, clips_input):
        """Parse clips input from command line format."""
        clips = []
        for clip_str in clips_input:
            try:
                if ':' in clip_str:
                    start_frame, duration = clip_str.split(':')
                    start_frame = int(start_frame)
                    duration = float(duration)
                    clips.append((start_frame, duration))
                else:
                    self.logger.error(f"Invalid clip format: {clip_str}")
                    return None
            except ValueError:
                self.logger.error(f"Invalid clip format: {clip_str}")
                return None
        return clips

    def interactive_extraction(self, video_path, video_info):
        """Interactive mode for defining and extracting clips."""
        print("\n🛠️  INTERACTIVE CLIP EXTRACTION")
        print("=" * 50)
        
        clips = []
        outputs = []
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        print("Enter clips one by one. Press Enter with empty input to finish.")
        print("You can specify start as frame number OR time (HH:MM:SS, MM:SS, or SS)")
        print()
        
        clip_num = 1
        while True:
            print(f"--- Clip {clip_num} ---")
            
            # Get start point
            while True:
                start_input = input(f"Start (frame number or time): ").strip()
                if not start_input:
                    if clips:
                        break
                    else:
                        print("Please add at least one clip.")
                        continue
                
                # Try to parse as frame number first
                try:
                    start_frame = int(start_input)
                    if 0 <= start_frame < total_frames:
                        break
                    else:
                        print(f"Frame number must be between 0 and {total_frames-1}")
                        continue
                except ValueError:
                    # Try to parse as time
                    start_frame = self.time_to_frames(start_input, fps)
                    if start_frame is None or start_frame >= total_frames:
                        print("Invalid time format. Use HH:MM:SS, MM:SS, or SS")
                        continue
                    break
            
            if not start_input:  # User pressed enter to finish
                break
            
            # Get duration
            while True:
                duration_input = input("Duration in seconds: ").strip()
                try:
                    duration = float(duration_input)
                    if duration > 0:
                        break
                    else:
                        print("Duration must be positive")
                except ValueError:
                    print("Please enter a valid number")
            
            # Get output name (optional)
            output_name = input("Output filename (optional): ").strip()
            if not output_name:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_name = f"{base_name}_clip_{clip_num:02d}_frame_{start_frame}_duration_{duration}s.mp4"
            
            clips.append((start_frame, duration))
            outputs.append(output_name)
            
            start_time = self.frames_to_time(start_frame, fps)
            print(f"✓ Added: Frame {start_frame} ({start_time}) for {duration}s -> {output_name}")
            print()
            
            clip_num += 1
        
        if not clips:
            print("No clips to extract.")
            return
        
        # Extract all clips
        print(f"\n🚀 EXTRACTING {len(clips)} CLIPS...")
        print("=" * 50)
        
        success_count = 0
        for i, ((start_frame, duration), output_name) in enumerate(zip(clips, outputs), 1):
            print(f"\nClip {i}/{len(clips)}: {output_name}")
            success = self.extract_clip(video_path, output_name, start_frame, duration)
            
            if success:
                success_count += 1
                print(f"✓ Successfully created: {output_name}")
            else:
                print(f"✗ Failed to create: {output_name}")
        
        print(f"\n🎉 EXTRACTION COMPLETE: {success_count}/{len(clips)} clips successful")

    def command_line_extraction(self, video_path, clips_input, outputs_input=None):
        """Extract clips from command line arguments."""
        clips = self.parse_clips_input(clips_input)
        if not clips:
            return False
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        success_count = 0
        
        print(f"\n🚀 EXTRACTING {len(clips)} CLIPS...")
        print("=" * 50)
        
        for i, (start_frame, duration) in enumerate(clips):
            # Determine output name
            if outputs_input and i < len(outputs_input):
                output_name = outputs_input[i]
            else:
                output_name = f"{base_name}_clip_{i+1:02d}_frame_{start_frame}_duration_{duration}s.mp4"
            
            print(f"\nClip {i+1}/{len(clips)}: {output_name}")
            print(f"  Start frame: {start_frame}, Duration: {duration}s")
            
            success = self.extract_clip(video_path, output_name, start_frame, duration)
            
            if success:
                success_count += 1
                print(f"✓ Successfully created: {output_name}")
            else:
                print(f"✗ Failed to create: {output_name}")
        
        print(f"\n🎉 EXTRACTION COMPLETE: {success_count}/{len(clips)} clips successful")
        return success_count == len(clips)

def main():
    parser = argparse.ArgumentParser(description='Complete Video Clipper - Analysis and Extraction')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--clips', nargs='+', help='Clips in format "start_frame:duration" (e.g., --clips 0:30 900:15)')
    parser.add_argument('--outputs', nargs='+', help='Custom output names for clips (optional)')
    parser.add_argument('--info-only', action='store_true', help='Show video information only')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level.upper())
    
    # Initialize video clipper
    clipper = CompleteVideoClipper(log_level=log_level)
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        clipper.logger.error(f"Input video file not found: {args.input_video}")
        print(f"Error: Video file '{args.input_video}' not found")
        return
    
    # Get video information
    print("Analyzing video file...")
    video_info = clipper.get_video_info(args.input_video)
    
    if not video_info:
        print("Failed to analyze video file")
        return
    
    # Display video information
    clipper.display_video_info(video_info, args.input_video)
    
    # Info only mode
    if args.info_only:
        return
    
    # Command line clips mode
    if args.clips:
        clipper.command_line_extraction(args.input_video, args.clips, args.outputs)
        return
    
    # Interactive mode (default)
    clipper.interactive_extraction(args.input_video, video_info)

if __name__ == "__main__":
    main()



    # step 1) python video_splitter_improve.py input_video.mp4 --info-only
    #step 2) python video_splitter_improve.py input_video.mp4 --clips 0:30 900:15 1800:45 --outputs intro.mp4 middle_scene.mp4 finale.mp4