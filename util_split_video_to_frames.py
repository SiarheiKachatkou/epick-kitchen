import os
import glob
import cv2
from utils_snippets import clip_to_frame_list
from consts import height,width,segment_count,class_counts, repo, batch_size, nouns, verbs


if __name__=="__main__":
    src_root='data/clips_b'

    src_video_paths = glob.glob(src_root+'/*')

    for i, src_video_path in enumerate(src_video_paths):
        print(f'video_path={src_video_path}')
        frame_list = clip_to_frame_list(src_video_path, height, width)
        dst_dir=f'{src_root}_frames/{i}'
        os.makedirs(dst_dir)
        for f_idx,f in enumerate(frame_list):
            f=cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_dir,str(f_idx)+'.jpg'),f)


