import cv2
import os

def create_video_from_frames(frame_folder, output_video):
    frame_files = sorted(os.listdir(frame_folder))

    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()
    
create_video_from_frames('processedframes', 'outputvideo/video.mp4')