import cv2
import os
import sys
import logging
from datetime import datetime
import subprocess
import json
import shutil

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/orientation_fixer_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

def check_ffmpeg_availability():
    """
    Check if FFmpeg is available in the system.
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    logger = logging.getLogger(__name__)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        logger.info("FFmpeg is available")
        logger.debug(f"FFmpeg version info: {result.stdout.split()[2] if len(result.stdout.split()) > 2 else 'Unknown'}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"FFmpeg not available: {e}")
        return False

def get_video_info(video_path):
    """
    Get comprehensive video information including rotation.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Video information including rotation, dimensions, fps, etc.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Getting video info for: {video_path}")
    
    info = {
        'rotation': 0,
        'width': 0,
        'height': 0,
        'fps': 0,
        'duration': 0,
        'codec': 'unknown'
    }
    
    try:
        # Use ffprobe to get comprehensive video metadata
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', video_path
        ]
        
        logger.debug(f"Running ffprobe command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        
        if 'streams' not in metadata or not metadata['streams']:
            logger.warning("No streams found in video metadata")
            return info
        
        # Find video stream
        video_stream = None
        for stream in metadata['streams']:
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            logger.warning("No video stream found")
            return info
        
        # Extract basic info
        info['width'] = video_stream.get('width', 0)
        info['height'] = video_stream.get('height', 0)
        info['fps'] = eval(video_stream.get('r_frame_rate', '0/1'))  # Convert fraction to float
        info['codec'] = video_stream.get('codec_name', 'unknown')
        
        if 'format' in metadata and 'duration' in metadata['format']:
            info['duration'] = float(metadata['format']['duration'])
        
        logger.info(f"Video info - Resolution: {info['width']}x{info['height']}, FPS: {info['fps']:.2f}, Codec: {info['codec']}")
        
        # Detect rotation - try multiple methods
        rotation = 0
        
        # Method 1: Check tags for rotate
        if 'tags' in video_stream:
            for key, value in video_stream['tags'].items():
                if key.lower() in ['rotate', 'rotation']:
                    try:
                        rotation = int(float(value))
                        logger.info(f"Found rotation in tags.{key}: {rotation}°")
                        break
                    except (ValueError, TypeError):
                        continue
        
        # Method 2: Check side_data_list for display matrix
        if rotation == 0 and 'side_data_list' in video_stream:
            logger.debug(f"Checking {len(video_stream['side_data_list'])} side data entries")
            for side_data in video_stream['side_data_list']:
                side_data_type = side_data.get('side_data_type', '')
                logger.debug(f"Side data type: {side_data_type}")
                
                if 'Display Matrix' in side_data_type or 'displaymatrix' in side_data_type.lower():
                    if 'rotation' in side_data:
                        try:
                            rotation = -int(float(side_data['rotation']))  # FFmpeg rotation is often negative
                            logger.info(f"Found rotation in display matrix: {rotation}°")
                            break
                        except (ValueError, TypeError):
                            continue
        
        # Normalize rotation
        if rotation != 0:
            original_rotation = rotation
            rotation = rotation % 360
            if rotation < 0:
                rotation += 360
            
            # Only keep standard rotations
            if rotation not in [0, 90, 180, 270]:
                logger.warning(f"Non-standard rotation {rotation}°, rounding to nearest 90°")
                rotation = round(rotation / 90) * 90 % 360
            
            if original_rotation != rotation:
                logger.info(f"Normalized rotation from {original_rotation}° to {rotation}°")
        
        info['rotation'] = rotation
        logger.info(f"Final detected rotation: {rotation}°")
        
        return info
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe command failed: {e}")
        logger.error(f"ffprobe stderr: {e.stderr}")
        
        # Fallback to OpenCV for basic info
        logger.info("Falling back to OpenCV for video info")
        return get_video_info_opencv(video_path)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe JSON output: {e}")
        return get_video_info_opencv(video_path)
    except Exception as e:
        logger.error(f"Unexpected error getting video info: {e}")
        return get_video_info_opencv(video_path)

def get_video_info_opencv(video_path):
    """
    Fallback method to get video info using OpenCV.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Basic video information
    """
    logger = logging.getLogger(__name__)
    logger.info("Getting video info using OpenCV fallback")
    
    info = {
        'rotation': 0,  # OpenCV can't detect rotation metadata
        'width': 0,
        'height': 0,
        'fps': 0,
        'duration': 0,
        'codec': 'unknown'
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video with OpenCV: {video_path}")
            return info
        
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if info['fps'] > 0:
            info['duration'] = frame_count / info['fps']
        
        cap.release()
        
        logger.info(f"OpenCV video info - Resolution: {info['width']}x{info['height']}, FPS: {info['fps']:.2f}")
        logger.warning("Could not detect rotation with OpenCV - assuming 0°")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get video info with OpenCV: {e}")
        return info

def fix_video_orientation_ffmpeg(input_path, output_path, video_info):
    """
    Fix video orientation using FFmpeg with high quality preservation.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path for output video
        video_info (dict): Video information including rotation
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Using FFmpeg method for orientation fix ===")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    rotation = video_info.get('rotation', 0)
    logger.info(f"Current rotation: {rotation}°")
    
    if rotation == 0:
        logger.info("No rotation correction needed")
        if input_path != output_path:
            logger.info("Copying file without modification")
            shutil.copy2(input_path, output_path)
            return True
        return True
    
    try:
        # Build FFmpeg command with high quality settings
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        # Add rotation filter
        video_filters = []
        
        if rotation == 90:
            video_filters.append('transpose=1')  # 90° clockwise
            logger.info("Applying 90° clockwise transpose")
        elif rotation == 180:
            video_filters.append('hflip,vflip')  # 180° rotation
            logger.info("Applying 180° rotation (hflip,vflip)")
        elif rotation == 270:
            video_filters.append('transpose=2')  # 90° counter-clockwise
            logger.info("Applying 270° clockwise (90° counter-clockwise) transpose")
        
        # Apply video filters
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        
        # High quality encoding settings
        cmd.extend([
            '-c:v', 'libx264',           # Use H.264 codec
            '-preset', 'medium',         # Balance between speed and compression
            '-crf', '18',                # High quality (lower = higher quality)
            '-c:a', 'aac',               # Use AAC for audio
            '-b:a', '192k',              # Audio bitrate
            '-movflags', '+faststart',   # Optimize for web streaming
            '-metadata:s:v:0', 'rotate=0',  # Remove rotation metadata
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            output_path
        ])
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Execute FFmpeg with progress monitoring
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            universal_newlines=True
        )
        
        # Monitor progress
        stderr_output = []
        duration = video_info.get('duration', 0)
        
        while True:
            line = process.stderr.readline()
            if not line:
                break
            
            stderr_output.append(line.strip())
            
            # Parse progress from FFmpeg output
            if 'time=' in line and duration > 0:
                try:
                    time_str = line.split('time=')[1].split()[0]
                    if ':' in time_str:
                        time_parts = time_str.split(':')
                        current_time = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
                        progress = (current_time / duration) * 100
                        if progress <= 100:
                            logger.info(f"Progress: {progress:.1f}% ({current_time:.1f}s / {duration:.1f}s)")
                except:
                    pass
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            # Verify output file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"FFmpeg completed successfully")
                logger.info(f"Output file: {output_path} ({file_size:,} bytes)")
                
                # Verify the fix
                output_info = get_video_info(output_path)
                logger.info(f"Verification: Output video rotation is now {output_info['rotation']}°")
                logger.info(f"Output resolution: {output_info['width']}x{output_info['height']}")
                
                return True
            else:
                logger.error("FFmpeg reported success but output file not found")
                return False
        else:
            logger.error(f"FFmpeg failed with return code: {return_code}")
            logger.error("FFmpeg error output:")
            for line in stderr_output[-10:]:  # Show last 10 lines of error
                logger.error(f"  {line}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in FFmpeg orientation fix: {e}")
        return False

def fix_video_orientation_opencv(input_path, output_path, video_info):
    """
    Fix video orientation using OpenCV with quality preservation.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path for output video
        video_info (dict): Video information including rotation
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Using OpenCV method for orientation fix ===")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    rotation = video_info.get('rotation', 0)
    logger.info(f"Applying rotation correction: {rotation}°")
    
    if rotation == 0:
        logger.info("No rotation correction needed")
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return True
    
    try:
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open input video: {input_path}")
            return False
        
        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Input video properties:")
        logger.info(f"  Resolution: {original_width}x{original_height}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Total frames: {total_frames}")
        
        # Calculate output dimensions
        if rotation in [90, 270]:
            output_width, output_height = original_height, original_width
            logger.info(f"Output dimensions (rotated): {output_width}x{output_height}")
        else:
            output_width, output_height = original_width, original_height
        
        # Setup video writer with high quality
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            logger.error(f"Could not create output video writer: {output_path}")
            cap.release()
            return False
        
        logger.info("Starting frame-by-frame processing...")
        
        frame_count = 0
        last_progress = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply rotation correction
            if rotation == 90:
                corrected_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                corrected_frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                corrected_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                corrected_frame = frame
            
            out.write(corrected_frame)
            frame_count += 1
            
            # Progress logging (every 5%)
            if total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                if progress >= last_progress + 5:
                    logger.info(f"Progress: {progress}% ({frame_count}/{total_frames} frames)")
                    last_progress = progress
        
        # Cleanup
        cap.release()
        out.release()
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"OpenCV processing completed successfully")
            logger.info(f"Output file: {output_path} ({file_size:,} bytes)")
            logger.info(f"Processed frames: {frame_count}")
            return True
        else:
            logger.error("OpenCV processing completed but output file not found")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in OpenCV orientation fix: {e}")
        return False

def fix_video_orientation(input_path, output_path=None, method='auto'):
    """
    Main function to fix video orientation with automatic method selection.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path for output video (auto-generated if None)
        method (str): 'auto', 'ffmpeg', or 'opencv'
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING VIDEO ORIENTATION FIX")
    logger.info("=" * 60)
    
    # Validate input
    if not os.path.exists(input_path):
        logger.error(f"Input video file not found: {input_path}")
        return False
    
    file_size = os.path.getsize(input_path)
    logger.info(f"Input file: {input_path} ({file_size:,} bytes)")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        extension = os.path.splitext(input_path)[1]
        output_path = f"{base_name}_fixed{extension}"
        logger.info(f"Auto-generated output path: {output_path}")
    
    # Get video information
    logger.info("Analyzing video...")
    video_info = get_video_info(input_path)
    
    if video_info['rotation'] == 0:
        logger.info("✅ Video orientation is already correct (0° rotation)")
        if input_path != output_path:
            logger.info("Copying original file to output location")
            shutil.copy2(input_path, output_path)
        return True
    
    logger.info(f"🔄 Video needs rotation correction: {video_info['rotation']}°")
    
    # Select method
    if method == 'auto':
        if check_ffmpeg_availability():
            method = 'ffmpeg'
            logger.info("Auto-selected method: FFmpeg (recommended)")
        else:
            method = 'opencv'
            logger.info("Auto-selected method: OpenCV (FFmpeg not available)")
    
    # Execute fix
    success = False
    start_time = datetime.now()
    
    if method.lower() == 'ffmpeg':
        if not check_ffmpeg_availability():
            logger.error("FFmpeg method requested but FFmpeg is not available")
            return False
        success = fix_video_orientation_ffmpeg(input_path, output_path, video_info)
    elif method.lower() == 'opencv':
        success = fix_video_orientation_opencv(input_path, output_path, video_info)
    else:
        logger.error(f"Unknown method: {method}. Use 'auto', 'ffmpeg', or 'opencv'")
        return False
    
    # Report results
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    if success:
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        logger.info("✅ VIDEO ORIENTATION FIX COMPLETED SUCCESSFULLY")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Input size: {file_size:,} bytes")
        logger.info(f"Output size: {output_size:,} bytes")
        logger.info(f"Output file: {output_path}")
    else:
        logger.error("❌ VIDEO ORIENTATION FIX FAILED")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info("=" * 60)
    
    return success

def main():
    """Main function for command line usage."""
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        print("Video Orientation Fixer")
        print("=" * 50)
        print("Usage:")
        print("  python orientation_fixer.py input_video [output_video] [method]")
        print("")
        print("Examples:")
        print("  python orientation_fixer.py video.mp4")
        print("  python orientation_fixer.py video.mp4 fixed_video.mp4")
        print("  python orientation_fixer.py video.mp4 fixed_video.mp4 ffmpeg")
        print("  python orientation_fixer.py video.mp4 fixed_video.mp4 opencv")
        print("  python orientation_fixer.py video.mp4 fixed_video.mp4 auto")
        print("")
        print("Methods:")
        print("  auto    - Automatically select best method (default)")
        print("  ffmpeg  - Fast, high quality (recommended)")
        print("  opencv  - Frame-by-frame processing (slower)")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else 'auto'
    
    logger.info("Command line arguments:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path if output_path else 'Auto-generated'}")
    logger.info(f"  Method: {method}")
    
    success = fix_video_orientation(input_path, output_path, method)
    
    if success:
        print("\n✅ Video orientation fixed successfully!")
    else:
        print("\n❌ Failed to fix video orientation. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()