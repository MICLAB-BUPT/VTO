import os
import os.path as osp
import time
import sys
from track import *
from segment import *
from recognise import *


class gait_recognition:
    def __init__(self, device):
        self.device = device
    def inference(self, gallery_video_path, probe_video_path):
        output_dir = "./OutputVideos/"
        os.makedirs(output_dir, exist_ok=True)
        current_time = time.localtime()
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        video_save_folder = osp.join(output_dir, timestamp)
        os.makedirs(video_save_folder, exist_ok=True)

        # tracking
        print('gallery tracking...')
        gallery_track_result = track(gallery_video_path, self.device)
        print('probe tracking...')
        probe_track_result  = track(probe_video_path, self.device)
        
        print('gallery segmenting...')
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, './GaitSilhouette/')
        print('probe segmenting...')
        probe_silhouette  = seg(probe_video_path , probe_track_result, './GaitSilhouette/')

        # recognise
        gallery_feat = extract_sil(gallery_silhouette)
        probe1_feat  = extract_sil(probe_silhouette)

        gallery_probe_result, scores = compare(probe1_feat, gallery_feat)

        # write the result back to the video
        print('getting result...')
        img_list = writeresult(gallery_probe_result, gallery_video_path, self.device)
        for i, img in enumerate(img_list):
            cv2.imwrite(os.path.join(video_save_folder, f'result{i+1}.jpg'), img)

        return gallery_probe_result, img_list, scores


if __name__ == "__main__":
    gr = gait_recognition(torch.device('cuda'))
    result_score = []
    import fnmatch
    for i in range(115):
        print('number', i+1)
        path = os.path.join('/home/bupt1/qyh/test_dataset', str(i+1))
        for dp, dn, filenames in os.walk(path):
            file_name = fnmatch.filter(filenames, '*.mp4')
            file_name.remove('probe.mp4')
            gallery_name = file_name[0]
            print(gallery_name)
        gallery_video_path = os.path.join('/home/bupt1/qyh/test_dataset', str(i+1), gallery_name)
        probe_video_path = os.path.join('/home/bupt1/qyh/test_dataset', str(i+1), 'probe.mp4')
        result, imgs, scores = gr.inference(gallery_video_path, probe_video_path)
        result_score.append(list(scores.values()))
    print(result_score)
    import numpy as np
    import pandas as pd
    a = np.array(result_score)
    df = pd.DataFrame(a)
    df.to_csv('/home/bupt1/qyh/Gait-recognition/result1.csv', index = None, header = None)

        