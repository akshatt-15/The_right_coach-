import cv2
import os
import sys
import logging
from datetime import datetime

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
    log_filename = f"logs/frame_extractor_{timestamp}.log"
    
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

def extract_frame_at_timestamp(video_path, timestamp, output_path=None, max_adjustment_frames=2):
    """
    Extract a frame from a video at the specified timestamp with smart frame adjustment.
    
    Args:
        video_path (str): Path to the input video file
        timestamp (float): Timestamp in seconds where to extract the frame
        output_path (str, optional): Path to save the extracted frame
        max_adjustment_frames (int): Maximum number of frames to adjust if exact frame is unavailable
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting frame extraction from video: {video_path}")
    logger.info(f"Target timestamp: {timestamp}s")
    logger.info(f"Max frame adjustment: ±{max_adjustment_frames} frames")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        print(f"Error: Video file '{video_path}' not found.")
        return False
    
    logger.info(f"Video file exists: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        print(f"Error: Could not open video file '{video_path}'.")
        return False
    
    logger.info("Video file opened successfully")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties - Duration: {duration:.2f}s, FPS: {fps:.2f}, "
                f"Total frames: {total_frames}, Resolution: {width}x{height}")
    
    print(f"Video Info:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    
    # Check if timestamp is valid
    if timestamp < 0 or timestamp > duration:
        logger.error(f"Timestamp {timestamp}s is out of range (0 - {duration:.2f}s)")
        print(f"Error: Timestamp {timestamp}s is out of range (0 - {duration:.2f}s)")
        cap.release()
        return False
    
    # Calculate frame number for the given timestamp
    target_frame = int(timestamp * fps)
    logger.info(f"Target frame number: {target_frame}")
    
    # Try to extract frame with adjustment if needed
    frame_extracted = False
    actual_frame_used = target_frame
    
    # Try exact frame first
    logger.info(f"Attempting to extract frame at exact position: {target_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    
    if ret and frame is not None:
        logger.info("Successfully extracted frame at exact timestamp")
        frame_extracted = True
    else:
        logger.warning(f"Could not extract frame at exact position {target_frame}")
        logger.info("Attempting frame adjustment...")
        
        # Try adjusting frames forward and backward
        for adjustment in range(1, max_adjustment_frames + 1):
            # Try frames after target
            for direction, sign in [("after", 1), ("before", -1)]:
                adjusted_frame = target_frame + (sign * adjustment)
                
                if 0 <= adjusted_frame < total_frames:
                    logger.info(f"Trying frame {adjusted_frame} ({adjustment} frames {direction} target)")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, adjusted_frame)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        actual_frame_used = adjusted_frame
                        actual_timestamp = adjusted_frame / fps
                        logger.info(f"Successfully extracted frame at position {adjusted_frame} "
                                   f"(timestamp: {actual_timestamp:.3f}s)")
                        print(f"Note: Used frame at {actual_timestamp:.3f}s "
                              f"({adjustment} frames {direction} target)")
                        frame_extracted = True
                        break
                else:
                    logger.debug(f"Skipping frame {adjusted_frame} (out of bounds)")
            
            if frame_extracted:
                break
    
    if not frame_extracted:
        logger.error(f"Could not extract any frame near timestamp {timestamp}s "
                    f"(tried ±{max_adjustment_frames} frames)")
        print(f"Error: Could not extract frame at or near timestamp {timestamp}s")
        cap.release()
        return False
    
    # Generate output filename if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        actual_timestamp = actual_frame_used / fps
        output_path = f"{base_name}_frame_at_{actual_timestamp:.3f}s.jpg"
        logger.info(f"Auto-generated output filename: {output_path}")
    
    logger.info(f"Saving frame to: {output_path}")
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    
    if success:
        file_size = os.path.getsize(output_path)
        logger.info(f"Frame saved successfully - File: {output_path}, Size: {file_size} bytes")
        print(f"Frame extracted successfully and saved as: {output_path}")
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]} pixels")
        print(f"File size: {file_size} bytes")
    else:
        logger.error(f"Failed to save frame to {output_path}")
        print(f"Error: Could not save frame to {output_path}")
    
    # Release the video capture object
    cap.release()
    logger.info("Video capture released")
    
    return success

def parse_timestamp(timestamp_str):
    """
    Parse timestamp string in various formats (seconds, MM:SS, HH:MM:SS)
    
    Args:
        timestamp_str (str): Timestamp string
    
    Returns:
        float: Timestamp in seconds
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing timestamp: '{timestamp_str}'")
    
    timestamp_str = timestamp_str.strip()
    
    # If it's just a number (seconds)
    try:
        result = float(timestamp_str)
        logger.info(f"Parsed as seconds: {result}")
        return result
    except ValueError:
        logger.debug("Not a simple float, trying time format")
    
    # If it's in MM:SS or HH:MM:SS format
    parts = timestamp_str.split(':')
    
    if len(parts) == 2:  # MM:SS
        try:
            minutes, seconds = map(float, parts)
            result = minutes * 60 + seconds
            logger.info(f"Parsed MM:SS format: {minutes}:{seconds} = {result}s")
            return result
        except ValueError:
            logger.error(f"Invalid MM:SS format: {timestamp_str}")
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    
    elif len(parts) == 3:  # HH:MM:SS
        try:
            hours, minutes, seconds = map(float, parts)
            result = hours * 3600 + minutes * 60 + seconds
            logger.info(f"Parsed HH:MM:SS format: {hours}:{minutes}:{seconds} = {result}s")
            return result
        except ValueError:
            logger.error(f"Invalid HH:MM:SS format: {timestamp_str}")
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    
    else:
        logger.error(f"Unrecognized timestamp format: {timestamp_str}")
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")

def main():
    """Main function to handle command line usage or interactive input."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("=== Video Frame Extractor Started ===")
    
    if len(sys.argv) >= 3:
        # Command line usage
        logger.info("Running in command line mode")
        video_path = sys.argv[1]
        timestamp_str = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        max_adjustment = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        
        logger.info(f"Command line arguments - Video: {video_path}, "
                   f"Timestamp: {timestamp_str}, Output: {output_path}, "
                   f"Max adjustment: {max_adjustment}")
        
        try:
            timestamp = parse_timestamp(timestamp_str)
            success = extract_frame_at_timestamp(video_path, timestamp, output_path, max_adjustment)
            
            if success:
                logger.info("Frame extraction completed successfully")
            else:
                logger.error("Frame extraction failed")
                sys.exit(1)
                
        except ValueError as e:
            logger.error(f"Timestamp parsing error: {e}")
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        # Interactive mode
        logger.info("Running in interactive mode")
        print("=== Video Frame Extractor ===")
        print()
        
        # Get video path
        video_path = input("Enter the path to your video file: ").strip().strip('"\'')
        logger.info(f"User input - Video path: {video_path}")
        
        if not video_path:
            logger.error("No video path provided by user")
            print("Error: No video path provided.")
            return
        
        # Get timestamp
        print("\nEnter timestamp in one of these formats:")
        print("  - Seconds: 45.5")
        print("  - MM:SS: 1:30")
        print("  - HH:MM:SS: 0:1:30")
        
        timestamp_str = input("Timestamp: ").strip()
        logger.info(f"User input - Timestamp: {timestamp_str}")
        
        try:
            timestamp = parse_timestamp(timestamp_str)
        except ValueError as e:
            logger.error(f"Invalid timestamp from user: {e}")
            print(f"Error: {e}")
            return
        
        # Get output path (optional)
        output_path = input("Output filename (press Enter for auto naming): ").strip()
        logger.info(f"User input - Output path: {output_path if output_path else 'Auto-generated'}")
        if not output_path:
            output_path = None
        
        # Get max adjustment frames (optional)
        adjustment_input = input("Max frame adjustment (default 2, press Enter to use default): ").strip()
        max_adjustment = 2
        if adjustment_input:
            try:
                max_adjustment = int(adjustment_input)
                logger.info(f"User input - Max adjustment: {max_adjustment}")
            except ValueError:
                logger.warning(f"Invalid adjustment value '{adjustment_input}', using default: 2")
                print("Invalid adjustment value, using default (2)")
        
        print("\nProcessing...")
        logger.info("Starting frame extraction process")
        
        success = extract_frame_at_timestamp(video_path, timestamp, output_path, max_adjustment)
        
        if success:
            logger.info("Interactive session completed successfully")
        else:
            logger.error("Interactive session failed")

if __name__ == "__main__":
    main()