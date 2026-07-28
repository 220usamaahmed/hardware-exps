
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# --- CONFIGURATION ---
BAG_PATH = Path('/home/shokry/ur3e-trajectories/exports/no_critic')  # Path to your ROS 2 bag folder
TOPIC_NAME = '/zed/zed_node/depth/depth_registered'
FPS = 30  # Adjust to match your recording framerate
# ---------------------

def quantize_depth_upper_numpy_batch(
    depth_batch,
    step=0.1,
    min_value=None,
    max_value=None,
    preserve_zero=True,
):
    """
    Quantize a batch of depth images by rounding up to the nearest step.

    Examples:
        0.71 -> 0.8
        0.70 -> 0.7
        0.65 -> 0.7
        0.60 -> 0.6

    Supports shapes:
        (B, H, W)
        (B, 1, H, W)
        (B, T, 1, H, W)

    Args:
        depth_batch: numpy array
        step: quantization step
        min_value: optional minimum clipping value
        max_value: optional maximum clipping value
        preserve_zero: keep zero values as zero

    Returns:
        Quantized depth batch with the same shape.
    """

    depth_q = np.asarray(depth_batch).astype(np.float32).copy()

    if preserve_zero:
        zero_mask = depth_q == 0

    # Optional clipping
    if min_value is not None or max_value is not None:
        if min_value is None:
            min_value = np.min(depth_q)
        if max_value is None:
            max_value = np.max(depth_q)

        depth_q = np.clip(depth_q, min_value, max_value)

    # Small epsilon avoids changing exact bin values
    eps = 1e-6

    depth_q = np.ceil((depth_q - eps) / step) * step

    if preserve_zero:
        depth_q[zero_mask] = 0.0

    return depth_q
    

def process_frame(depth_img: np.ndarray) -> np.ndarray:
    depth_img = depth_img[60:170, 230:440]
    depth_img = np.nan_to_num(depth_img, nan=10.0)
    depth_img = np.clip(depth_img, 0, 0.8)
    
    first_box_start_x=41
    first_box_start_y=0
    first_box_end_x=110
    first_box_end_y=65

    second_box_start_x=41
    second_box_start_y=140
    second_box_end_x=110
    second_box_end_y=210        

    object_start_x=33
    object_start_y=98
    object_end_x=60
    object_end_y=125    
    
    first_drawer_start_x=0
    first_drawer_start_y=0
    first_drawer_end_x=41
    first_drawer_end_y=65
        
    second_drawer_start_x=0
    second_drawer_start_y=140
    second_drawer_end_x=41
    second_drawer_end_y=210
    
    depth_img = quantize_depth_upper_numpy_batch(depth_img, step=0.05)
    
    inbetween_region_first_box=depth_img[first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]
    inbetween_depth_value_first_box=np.percentile(inbetween_region_first_box,10)
    depth_img[first_box_start_x:first_box_end_x,first_box_start_y:first_box_end_y]=inbetween_depth_value_first_box
    
    inbetween_region_second_box=depth_img[second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]

    inbetween_depth_value_second_box=np.percentile(inbetween_region_second_box,10)
    depth_img[second_box_start_x:second_box_end_x,second_box_start_y:second_box_end_y]=inbetween_depth_value_second_box

    inbetween_region_first_drawer=depth_img[first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]

    inbetween_depth_value_first_drawer=np.percentile(inbetween_region_first_drawer,10)
    depth_img[first_drawer_start_x:first_drawer_end_x,first_drawer_start_y:first_drawer_end_y]=inbetween_depth_value_first_drawer
    
    inbetween_region_second_drawer=depth_img[second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]
    
    inbetween_depth_value_second_drawer=np.percentile(inbetween_region_second_drawer,10)
    depth_img[second_drawer_start_x:second_drawer_end_x,second_drawer_start_y:second_drawer_end_y]=inbetween_depth_value_second_drawer
    

    inbetween_region_object=depth_img[object_start_x:object_end_x,object_start_y:object_end_y]
    
    inbetween_depth_value_object=np.percentile(inbetween_region_object,10)
    depth_img[object_start_x:object_end_x,object_start_y:object_end_y]=inbetween_depth_value_object
                    
    return depth_img

def main():
    print(f"Opening bag: {BAG_PATH}")
    
    # Ask for video tag
    video_tag = input("Enter a video tag to add to the filename (or press Enter to skip): ").strip()
    if video_tag:
        output_mp4 = f'/home/shokry/ur3e-trajectories/exports/depth_video_{video_tag}.mp4'
    else:
        output_mp4 = '/home/shokry/ur3e-trajectories/exports/depth_video.mp4'
    
    video_writer = None
    
    # Create standard ROS 2 type store (handles Humble/Iron message schemas seamlessly)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    # AnyReader handles the opening and low-level deserialization for us
    with AnyReader([BAG_PATH], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == TOPIC_NAME]
        if not connections:
            print(f"Error: Topic '{TOPIC_NAME}' not found in bag.")
            return

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Let the reader handle the type decoding automatically
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            # Convert ROS 2 Image data array to a numpy array (32-bit float for depth)
            depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            
            # --- PROCESS FOR MP4 VISUALIZATION ---
            # 1. Clear out NaNs / Infs (lost camera frames)
            # depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 2. Normalize range (Assume max depth range of 2 meters for clipping)
            # max_dist = 2.0 
            # norm_img = np.clip(depth_img / max_dist, 0, 1) * 255
            # norm_img = norm_img.astype(np.uint8)
            
            img = process_frame(depth_img) * 255
            img = img.astype(np.uint8)
            
            # 3. Convert to grayscale (3-channel format for video writer)
            grayscale_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            print(f"Processed frame at timestamp: {timestamp} seconds")
            
            # --- SETUP WRITER ON FIRST FRAME ---
            if video_writer is None:
                height, width = grayscale_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPEG codec for lossless compression
                video_writer = cv2.VideoWriter(output_mp4, fourcc, FPS, (width, height))
                print(f"Exporting video at resolution: {width}x{height}...")

            video_writer.write(grayscale_frame)

    if video_writer:
        video_writer.release()
        print(f"Success! Video saved to {output_mp4}")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    main()
