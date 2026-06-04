
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# --- CONFIGURATION ---
BAG_PATH = Path('/home/shokry/ur3e-trajectories/exports/zed_depth_video')  # Path to your ROS 2 bag folder
TOPIC_NAME = '/zed/zed_node/depth/depth_registered'
FPS = 15  # Adjust to match your recording framerate
# ---------------------

def main():
    print(f"Opening bag: {BAG_PATH}")
    
    # Ask for video tag
    video_tag = input("Enter a video tag to add to the filename (or press Enter to skip): ").strip()
    if video_tag:
        output_mp4 = f'depth_video_{video_tag}.mp4'
    else:
        output_mp4 = 'depth_video.mp4'
    
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
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 2. Normalize range (Assume max depth range of 2 meters for clipping)
            max_dist = 2.0 
            norm_img = np.clip(depth_img / max_dist, 0, 1) * 255
            norm_img = norm_img.astype(np.uint8)
            
            # 3. Convert to grayscale (3-channel format for video writer)
            grayscale_frame = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
            
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
