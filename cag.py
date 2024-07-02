from datetime import datetime
import os
import pandas as pd
import numpy as np
from numpy.lib.function_base import disp
from skimage import exposure, img_as_ubyte

import torch
import decord
import cv2
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import pydicom


def normalize(pixel_array, window_center, window_width):
    # Calculate the min and max values for the window
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # Apply windowing
    windowed_array = np.clip(pixel_array, window_min, window_max)

    # Scale the pixel values to [0, 1]
    scaled_array = ((windowed_array - window_min) / (window_max - window_min))
    return scaled_array 


class CAGVideoMAE(torch.utils.data.Dataset):
   
    def __init__(self,
                 setting,
                 train=True,
                 test_mode=False,
                 transform=None):

        super(CAGVideoMAE, self).__init__()
        self.setting = setting  # The path of `train.csv` file.
        self.train = train
        self.test_mode = test_mode
        self.transform = transform
        self.clips = self._make_dataset(setting)
        print(f"Total number of train videos={len(self.clips)}")

    def __getitem__(self, index):
        # e.g. directory: data/snubhcvc/raw/cag_ccta_1yr_all/data/12693133/20170224/XA/001.dcm
        dcm_path= self.clips[index]
        dcm = pydicom.dcmread(dcm_path)

        # window_center = dcm.WindowCenter
        # window_width = dcm.WindowWidth
        pixel_array = dcm.pixel_array
        # print(f"[CAGVideoMAE.__getitem__] pixel_array.shape={pixel_array.shape}")

        """vjepa dataset configuration
            self.duration=None
            self.frames_per_clip=5
            self.frame_step=2
            self.filter_short_videos=False
            clip_len=10
            self.num_clips=1
            self.random_clip_sampling=True
            self.allow_clip_overlap=False
            
            
            vr.shape=(36, 512, 512, 1)
            [loadvideo_decord] start_indx=0, end_indx=10
            vr.shape=(58, 512, 512, 1)
            [loadvideo_decord] start_indx=22, end_indx=32
            vr.shape=(72, 512, 512, 1)
            [loadvideo_decord] start_indx=12, end_indx=22
            vr.shape=(47, 512, 512, 1)
            [loadvideo_decord] start_indx=8, end_indx=18
            vr.shape=(44, 512, 512, 1)
            [loadvideo_decord] start_indx=3, end_indx=13
            vr.shape=(50, 512, 512, 1)                                                                                        [loadvideo_decord] start_indx=14, end_indx=24
        """
        fpc = frames_per_clip = 6
        fstp = frame_step = 2
        num_clips = 1
        random_clip_sampling = True
        allow_clip_overlap = False

        clip_len = int((fpc - 1) * fstp)  # 10
        partition_len = len(pixel_array) // num_clips  # len(pixel_array)

        all_indices, clip_indices = [], []
        for i in range(num_clips):
            if partition_len > clip_len:
                """
                우리의 경우 
                    num_clips = 1
                    clip_len = 10
                이므로 partition_len == len(vr) 이다.
                따라서 dicom 프레임 개수가 10보다 크면 이 조건에 걸리게 되고
                그렇지 않으면 아래 else 조건에 걸리게 된다.
                """
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if random_clip_sampling:  # default값인 True로 사용중
                    end_indx = np.random.randint(clip_len + fstp, partition_len - fstp)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc).astype(np.int64)
                # indices = np.arange(start_indx, end_indx, fstp)
                # indices = np.clip(indices, 0, partition_len).astype(np.int64)
                # --
                indices = indices + i * partition_len
                # print(f"\t[CAGVideoMAE.__getitem__] indices={indices}")
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(pixel_array)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(pixel_array) > clip_len:
                        clip_step = (len(pixel_array) - clip_len) // (num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))
        
        buffer = pixel_array[all_indices]
        if buffer.dtype == np.uint16:
            buffer = img_as_ubyte(exposure.rescale_intensity(buffer))

        # ############################################################
        # timestamp = datetime.now().strftime("%H%M%S")
        # components = dcm_path.split("/")[-4:-1]
        # basename = dcm_path.split("/")[-1].split(".")[0]
        # components.append(basename)
        # components.append(timestamp)
        # save_path = "_".join(components) + ".png"

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(2, 3, figsize=(30, 20))
        # axes[0, 0].imshow(buffer[0])
        # axes[0, 1].imshow(buffer[1])
        # axes[0, 2].imshow(buffer[2])
        # axes[1, 0].imshow(buffer[3])
        # axes[1, 1].imshow(buffer[4])
        # axes[1, 2].imshow(buffer[5])
        # fig.tight_layout()
        # fig.savefig(save_path)
        # ############################################################

        rows = dcm.Rows
        if rows != 512:
            resized_imgs = []
            for frame in buffer:
                resized_imgs.append(cv2.resize(frame, (512, 512)))
            buffer = np.array(resized_imgs)
        process_data, mask = self.transform(buffer) # (C, T, H, W)
        # print("\n")
        # print(f"[CAGVideoMAE] process_data.shape={process_data.shape}")
        # print(f"[CAGVideoMAE] mask.shape={mask.shape}")
        # print("\n")
        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, csv_path):
        df = pd.read_csv(csv_path, header=None, delimiter=",")
        clips = df.values[:, 0]
        # np.random.shuffle(clips)
        # clips = clips[:100]
        return clips
