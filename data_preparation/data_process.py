import glob
import os
import subprocess

video_dir = "./FINADiving"
video_list = glob.glob(os.path.join(video_dir, "*.mp4"))
base_dir = "./FINADiving_jpgs"
save_dir = "./FINADiving_jpgs_256"

#################### step 1: video2frames ####################
for vid in video_list:
    vid_dir = vid.split("/")[-1][:-4]
    vid_dir_1 = os.path.join(base_dir, vid_dir)
    if not os.path.exists(vid_dir_1):
        os.mkdir(vid_dir_1)
    vid_dir_2 = vid_dir_1 + "/%05d.jpg"
    subprocess.call("ffmpeg -i %s -f image2 -qscale:v 2 %s" % (vid, vid_dir_2), shell=True)

#################### step 2: frames resize ####################
if base_dir is None:
    raise RuntimeError('please choose the base dir and save dir')
video_list = os.listdir(base_dir)
for video in video_list:
    save_path = os.path.join(save_dir, video)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frame_list = sorted(os.listdir(os.path.join(base_dir,video)))
    for frame in frame_list:
        subprocess.call('ffmpeg -i %s -vf scale=-1:256 %s' % (os.path.join(base_dir, video, frame), os.path.join(save_dir, video, frame)), shell=True)