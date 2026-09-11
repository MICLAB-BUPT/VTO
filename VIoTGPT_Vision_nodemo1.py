# coding: utf-8
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    OwlViTProcessor,
    OwlViTForObjectDetection,
    AutoModelForZeroShotObjectDetection,
)
import os

import sys

import random
import torch
import cv2
import shutil
import traceback
import numpy as np

sys.path.append("/home/wyt/VIoTGPT/tools/")
# 1
sys.path.append("/home/wyt/VIoTGPT/tools/reid-test/")
sys.path.append("/home/wyt/VIoTGPT/tools/reid-test/demo/")
from fastreid.config import get_cfg
from predictor import FeatureExtractionDemo
import torch.nn.functional as Fun

# 2
# sys.path.append("./tools/fire-smoke-detection/")
# sys.path.append("./tools/fire-smoke-detection/models")
from pathlib import Path
import queue
from FireSmokeDetection.experimental import attempt_load
from FireSmokeDetection.utils.datasets import LoadImages
from FireSmokeDetection.utils.general import (
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    plot_one_box,
)

# 3
import string
import shortuuid
from typing import List

import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 4

from mmpose.apis import MMPoseInferencer

# 5
from ultralytics import YOLO

# Child detection (DEIMv2-Wholebody34; ONNX Runtime is loaded lazily)
from ChildDetection import ChildDetection


# SceneRecognition
# sys.path.append('/home/wyt/VIoTGPT/tools/place/')
import pandas as pd
from torchvision import transforms
import place.wideresnet as wideresnet
from tqdm import *

# holmesvau
sys.path.append("/home/wyt/VIoTGPT/tools/HolmesVAU/")
from holmesvau.holmesvau_utils import load_model, generate

# CrowdCounting
sys.path.append("/home/wyt/VIoTGPT/tools/CLTR_crowdcounting/")

from collections import OrderedDict
import argparse
# Assuming config.py, utils, and Networks are in the './tools/crowd-counting/' directory

from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from CLTR_crowdcounting.utils import setup_seed
import util.misc as utils
from Networks.CDETR import build_model
import torch.nn as nn

# HumanPoseTracking
import os
import sys
import torch
import cv2
import yaml
from tqdm import tqdm
import numpy as np

# VadCLIP
import VadCLIP.src.clip as clip
from VadCLIP.src.model import CLIPVAD

script_directory = os.path.dirname(os.path.abspath(__file__))

motip_tool_path = "/home/wyt/VIoTGPT/tools/MOTIP_MultipleObjectTracking/"
sys.path.append(motip_tool_path)
from MOTIP_MultipleObjectTracking.models.motip import build as build_TrackingModel
from MOTIP_MultipleObjectTracking.models.misc import load_checkpoint
from MOTIP_MultipleObjectTracking.models.runtime_tracker import RuntimeTracker
from MOTIP_MultipleObjectTracking.utils.nested_tensor import (
    nested_tensor_from_tensor_list,
)
from MOTIP_MultipleObjectTracking.configs.util import load_super_config
from torchvision.transforms import functional as F

# GaitRecognition
sys.path.append('/home/wyt/VIoTGPT/tools/Gait-recognition/')
from track import *
from segment import *
from recognise import *

#  Face Detection Tool ---
sys.path.append("/home/wyt/VIoTGPT/tools/FaceDetection_DSFD/")
from FaceDetection_DSFD.face_ssd import build_ssd
from FaceDetection_DSFD.data import WIDERFace_CLASSES, widerface_640, TestBaseTransform
# from widerface_val import bbox_vote

# Low-Light Enhancement
import torchvision
import time
import ZeroDCE_LowLightEnhancement.ZeroDCE_code.model as LowLightEnhancer_model

# ImageSuperResolution
# from mmagic.apis import MMagicInferencer

# HumanFallDetection
from HumanFallDetection.fall_core import FallDetectorMulti
from HumanFallDetection.config import (
    FPS,
    WINDOW_SIZE,
    V_THRESH,
    DY_THRESH,
    ASPECT_RATIO_THRESH,
)

# Plate Recognition
sys.path.append('/home/wyt/VIoTGPT/tools/plate_recognition/')
from PIL import Image, ImageDraw, ImageFont
from rpnet.demo_plate import fh02

# Gait Recognition
sys.path.append('/home/wyt/VIoTGPT/project/Gait-recognition/')
from track import *
from segment import *
from recognise import *

PREFIX = """You are designed to assist with multimodal analysis for social safety governance, focusing on video and image surveillance.
    You cannot directly read images or videos, but it has a series of visual tools to accomplish different monitoring. 
    You can invoke different tools to indirectly understand the picture and the video indirectly. 
    You are very strict about filenames and will never fake nonexistent files. 
    You are able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. 
    For improved performance, You are designed to utilize a combination of tools on a single task, allowing it to establish and exploit the relationships between different tool outputs.

    TOOLS:
    ------

    You have access to the following tools:"""

# FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

# ```
# Thought: Do I need to use a tool? Yes.
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ```
# **IMPORTANT RULE: The 'Action Input' field must ONLY contain the required values (like file paths or text), separated by commas if necessary. NEVER include parameter names like 'image_path=', 'video_path=', or 'text_prompt=' in the Action Input.**
# When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format(the prefix of "Thought: " and "{ai_prefix}: " are must be included):

# ```
# Thought: Do I need to use a tool? No.
# {ai_prefix}: Final Answer: [your response here]
# ```
# """
FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
**IMPORTANT RULE: The 'Action Input' field must ONLY contain the required values (like file paths or text), separated by commas if necessary. NEVER include parameter names like 'image_path=', 'video_path=', or 'text_prompt=' in the Action Input.**
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format(the prefix of "Thought: " and "{ai_prefix}: " are must be included):

```
Thought: Do I need to use a tool? No.
{ai_prefix}: Final Answer: [your response here]
```
Below are a few examples demonstrating how to use tools correctly:
Example 1:
Human: provide a figure named ./Evo_data/images/image_3ff56c.jpg and a video named ./Evo_data/videos/video_53ad9a.mp4. Can you confirm if the person from the uploaded image has appeared in the video?
Thought: Do I need to use a tool? Yes.
Action: Recognize the Person by Appearance
Action Input: ./Evo_data/images/image_3ff56c.jpg,./Evo_data/videos/video_53ad9a.mp4
Observation: Video ./Evo_data/videos/video_53ad9a.mp4 have this identity. The person appeared in 12 frames, including Frame [45, 67, 89, 112, 145].
Thought: Do I need to use a tool? No.
Final Answer: AI: Yes, the person from the query image appears in the video, appearing in 12 frames such as Frame 45, 67, and 89.
Example 2:
Human: provide a video named ./Evo_data/videos/video_ebb1c7.mp4. Can you identify if there is smoke or flame in the video?
Thought: Do I need to use a tool? Yes.
Action: Fire and Smoke Detection
Action Input: ./Evo_data/videos/video_ebb1c7.mp4
Observation: Dangerous! We have detected both smoke and file in video ./Evo_data/videos/video_ebb1c7.mp4.
Thought: Do I need to use a tool? No.
AI: Final Answer: Yes, both smoke and fire have been detected in the video ./Evo_data/videos/video_ebb1c7.mp4, indicating a dangerous situation.
"""

SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since You are a text language model, you must use tools to observe images rather than imagination.
The thoughts and observations are only visible for you, you should remember to repeat important information in the final response for Human. 
Let's think step by step. {agent_scratchpad} 
"""

# os.makedirs("image", exist_ok=True)
os.makedirs("images", exist_ok=True)


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def visualize(image, faces, return_msg, thickness=2):
    input = image.copy()
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print(
                "Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}".format(
                    idx, face[0], face[1], face[2], face[3], face[-1]
                )
            )
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(
                input,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                (0, 255, 0),
                thickness,
            )
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(
        input,
        return_msg,
        (1, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        thickness,
    )
    return input


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = np.zeros((0, 5), dtype=np.float32)
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(
            det_accu[:, -1:]
        )
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets
class GaitRecognition:
    def __init__(self, device):
        print("Initializing Gait Recognition")
        self.output_dir = "videos"
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = device
    @prompts(name="Recognize the Person by Gait",
             description="useful when you want to know whether the person in the uploaded video appeared in the video. "
                         "The tool recognize people by gait, that is the way people walks or runs. "
                         "The input to this tool should be a comma separated string of two, representing the uploaded video_path and the target video_path. "
                         "The input must only contain the file paths, without parameter names like 'image_path=' or 'video_path='. For example, a valid input is './Evo_data/videos/videos_68a5b9.mp4,./Evo_data/videos/video_79c531.mp4'.")
    def inference(self, inputs):
        # probe_video_path, gallery_video_path = inputs.split(',')
        probe_video_path, gallery_video_path = [part.strip() for part in inputs.split(",", 1)]
        print(probe_video_path, gallery_video_path)
        if 'mp4' not in probe_video_path:
            final_msg = 'mp4 {} does not exist.'.format(probe_video_path)
            return final_msg
        # gallery_video_path = gallery_video_path.split('/')[0] + '/Gait_' + gallery_video_path.split('/')[-1]
        gallery_video = cv2.VideoCapture(gallery_video_path)
        probe_video = cv2.VideoCapture(probe_video_path)
        if not gallery_video.isOpened():
            final_msg = 'Video {} does not exist.'.format(gallery_video_path)
            return final_msg
        if not probe_video.isOpened():
            final_msg = 'Video {} does not exist.'.format(probe_video_path)
            return final_msg
        video_save_folder = 'images'
        os.makedirs(video_save_folder, exist_ok=True)
        # tracking
        print('gallery tracking...')
        gallery_track_result = track(gallery_video_path, self.device)
        print('probe tracking...')
        probe_track_result = track(probe_video_path, self.device)

        print('gallery segmenting...')
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, './GaitSilhouette/')
        print('probe segmenting...')
        probe_silhouette = seg(probe_video_path, probe_track_result, './GaitSilhouette/')
        # recognise
        gallery_feat = extract_sil(gallery_silhouette)
        probe1_feat = extract_sil(probe_silhouette)
        gallery_probe_result, scores = compare(probe1_feat, gallery_feat)
       # write the result back to the video
        print('getting result...')
        img_list = writeresult(gallery_probe_result, gallery_video_path, self.device)
        for i, img in enumerate(img_list):
            save_name = os.path.join(video_save_folder, f'result{i + 1}.jpg')
            cv2.imwrite(save_name, img)
        for key in scores:
            if scores[key] < 10.0:
                msg = 'According to the gait analysis, we have found this person. Please refer to {}. '.format(
                    save_name)
            else:
                msg = 'According to the gait analysis, we did not found this person.'
        return msg

class PlateRecognition:
    def __init__(self, device):
        self.device = device
        self.numClasses = 4
        self.img_size = (480, 480)
        self.resume_file = "/home/wyt/VIoTGPT/tools/plate_recognition/fh02.pth"
        self.provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
                          "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        self.alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
        self.ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        self.model = fh02()
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model.load_state_dict(torch.load(self.resume_file))
        self.model.cuda()
        self.model.eval()

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32')
        img /= 255.0
        img = np.expand_dims(img, 0)
        img = torch.tensor(img)
        return img

    @prompts(name="Recognize the Vehicle by Plate Number",
             description="useful when you want to know whether the vehicle of given vehicle plate appeared in the video. "
                         "The tool recognize the vehicle by the plate number."
                         "The input to this tool should be a comma separated string of two, representing the vehicle plate and the video_path. "
                         "The input must only contain the file paths, without parameter names like 'image_path=' or 'video_path='. For example, a valid input is '京A682K5,./Evo_data/videos/video_79c531.mp4'.")
    def inference(self, inputs):
        try:
            # query_plate, video_path = inputs.split(',')
            query_plate, video_path= [part.strip() for part in inputs.split(",", 1)]
            # video_path = video_path.split('/')[0] + '/PlateRecognition_' + video_path.split('/')[-1]
            video = cv2.VideoCapture(video_path)
            frame_cnt = 0
            if not video.isOpened():
                final_msg = 'Video {} does not exist.'.format(video_path)
                return final_msg
            while True:
                ret, frame = video.read()
                if frame is None:
                    final_msg = 'Vehicle plate {} does not appear in video {}'.format(query_plate, video_path)
                    return final_msg
                if ret is True:
                    frame_cnt += 1
                    img = self.preprocess(frame)
                    bbox, pred = self.model(img)
                    outputY = [el.data.cpu().numpy().tolist() for el in pred]
                    labelPred = [t[0].index(max(t[0])) for t in outputY]
                    [cx, cy, w, h] = bbox.data.cpu().numpy()[0].tolist()
                    lpn = self.provinces[labelPred[0]] + self.alphabets[labelPred[1]] + self.ads[labelPred[2]] + self.ads[
                        labelPred[3]] + self.ads[labelPred[4]] + self.ads[labelPred[5]] + self.ads[labelPred[6]]
                    if lpn == query_plate:
                        left_up = [(cx - w / 2) * frame.shape[1], (cy - h / 2) * frame.shape[0]]
                        right_down = [(cx + w / 2) * frame.shape[1], (cy + h / 2) * frame.shape[0]]
                        cv2.rectangle(frame, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])),
                                      (0, 0, 255), 2)
                        pilImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pilImg)
                        font = ImageFont.truetype("/home/wyt/VIoTGPT/tools/plate_recognition/rpnet/SimHei.ttf", 50, encoding='utf-8')
                        draw.text((int(left_up[0]), int(left_up[1]) - 40), lpn, (255, 0, 0),
                                  font=font)
                        cv2charimg = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
                        cv2.imwrite("output/{}.jpg".format(frame_cnt), cv2charimg)
                        print("Image saved at output/{}.jpg".format(frame_cnt))
                        print("Query plate found at frame {}".format(frame_cnt))
                        final_msg = 'Video {} have this vehicle. The vehicle appeared in {} frames'.format(video_path, frame_cnt)
                        return final_msg
        except:
            return "Unkown Error in PlateRecognition."

class VehicleReid:
    def __init__(self, device):
        print(f"Initializing Vehicle Re-Identification")
        self.cfg = get_cfg()
        self.cfg.merge_from_file("/home/wyt/VIoTGPT/tools/reid-test/demo/Base-SBS.yml")
        self.cfg.merge_from_list(
            [
                "MODEL.WEIGHTS",
                "/home/wyt/VIoTGPT/tools/reid-test/demo/models/veri_sbs_R50-ibn.pth",
            ]
        )
        self.cfg.freeze()
        self.demo = FeatureExtractionDemo(self.cfg, parallel=False)
        self.threshold = 0.95

    def postprocess(self, features):
        features = Fun.normalize(features)
        features = features.cpu().data.numpy()
        return features

    @prompts(
        name="Recognize the Vehicle by Appearance",
        description="useful when you want to know whether the vehicle in the photo appeared in the video. "
        "The tool recognize the vehicle by appearance. "
        "The input to this tool should be a comma separated string of two, representing the image_path and the video_path. "
        "The input must only contain the file paths, without parameter names like 'image_path=' or 'video_path='. For example, a valid input is './Evo_data/images/image_68a5b9.jpg,./Evo_data/videos/video_79c531.mp4'."
    )
    def inference(self, inputs):
        try:
            final_msg = "final_msg"
            # query_path, gallery_path = inputs.split(",")
            query_path, gallery_path = [part.strip() for part in inputs.split(",", 1)]
            if query_path is None:
                final_msg = "Cannot find a query image in {}".format(query_path)
                return final_msg
            query_img = cv2.imread(query_path)
            query_feat = self.demo.run_on_image(query_img)
            query_feat = self.postprocess(query_feat)
            # gallery_path = (
            #     gallery_path.split("/")[0]
            #     + "/VehicleReid_"
            #     + gallery_path.split("/")[-1]
            # )
            video = cv2.VideoCapture(gallery_path)
            frame_cnt = 0
            return_frame_id = []
            if not video.isOpened():
                final_msg = "Video {} does not exist.".format(gallery_path)
                return final_msg
            while True:
                ret, frame = video.read()
                if frame is None:
                    break
                if ret is True:
                    frame_cnt += 1
                    frame_feat = self.demo.run_on_image(frame)
                    frame_feat = self.postprocess(frame_feat)
                    cos_score = np.matmul(query_feat, frame_feat.T)
                    if cos_score >= self.threshold:
                        return_frame_id.append(frame_cnt)
            if len(return_frame_id) > 0 and len(return_frame_id) < 30:
                final_msg = "Video {} have this vehicle. The vehicle appeared in {} frames, including Frame {}.".format(
                    gallery_path, len(return_frame_id), return_frame_id
                )
            elif len(return_frame_id) >= 30:
                final_msg = "Video {} have this vehicle. The vehicle appeared in {} frames, including Frame {}, etc.".format(
                    gallery_path, len(return_frame_id), return_frame_id[:30]
                )
            else:
                final_msg = "Video {} does not have {}.".format(
                    gallery_path, "this vehicle"
                )
            print("final_msg: ", final_msg)
            return final_msg
        except:
            return "Unkown Error in VehicleReid."


class PersonReid:
    def __init__(self, device):
        print("Initializing Person Re-identification")
        self.cfg = get_cfg()
        self.cfg.merge_from_file("/home/wyt/VIoTGPT/tools/reid-test/demo/Base-SBS.yml")
        self.cfg.merge_from_list(
            [
                "MODEL.WEIGHTS",
                "/home/wyt/VIoTGPT/tools/reid-test/demo/models/market_sbs_R50-ibn.pth",
            ]
        )
        self.cfg.freeze()
        self.threshold = 0.975
        self.demo = FeatureExtractionDemo(self.cfg, parallel=False)

    def postprocess(self, features):
        features = Fun.normalize(features)
        features = features.cpu().data.numpy()
        return features

    @prompts(
        name="Recognize the Person by Appearance",
        description="useful when you want to know whether the person in the photo appeared in the video. "
        "The tool recognize people by appearance, that is body shape and clothing. "
        "The input to this tool should be a comma separated string of two, representing the image_path and the video_path."
        "The input must only contain the file paths, without parameter names like 'image_path=' or 'video_path='. For example, a valid input is './Evo_data/images/image_68a5b9.jpg,./Evo_data/videos/video_79c531.mp4'.",
    )
    def inference(self, inputs):
        try:
            query_path, gallery_path = [part.strip() for part in inputs.split(",", 1)]

            if not os.path.exists(query_path) or query_path is None:
                final_msg = "Cannot find a query image in {}".format(query_path)

                return final_msg
            query_img = cv2.imread(query_path)
            query_feat = self.demo.run_on_image(query_img)
            query_feat = self.postprocess(query_feat)
            # gallery_path = (
            #     gallery_path.split("/")[0]
            #     + "/PersonReid_"
            #     + gallery_path.split("/")[-1]
            # )

            video = cv2.VideoCapture(gallery_path)
            frame_cnt = 0
            return_frame_id = []
            if not video.isOpened():
                final_msg = "Video {} does not exist.".format(gallery_path)
                return final_msg
            while True:
                ret, frame = video.read()
                if frame is None:
                    break
                if ret is True:
                    frame_cnt += 1
                    frame_feat = self.demo.run_on_image(frame)
                    frame_feat = self.postprocess(frame_feat)
                    cos_score = np.matmul(query_feat, frame_feat.T)
                    if cos_score >= self.threshold:
                        return_frame_id.append(frame_cnt)
            if len(return_frame_id) > 0 and len(return_frame_id) < 30:
                final_msg = "Video {} have this identity. The person appeared in {} frames, including Frame {}.".format(
                    gallery_path, len(return_frame_id), return_frame_id
                )
            elif len(return_frame_id) >= 30:
                final_msg = "Video {} have this identity. The person appeared in {} frames, including Frame {}, etc.".format(
                    gallery_path, len(return_frame_id), return_frame_id[:30]
                )
            else:
                final_msg = "Video {} does not have {}.".format(
                    gallery_path, "this identity"
                )
            print(final_msg)
            return final_msg
        except:
            return "Unkown Error in PersonReid"


class ObjectLOCTool:  # <-- Inherit from nothing specific
    # model_path = "google/owlvit-base-patch32" # <-- Remove or move to __init__

    # device = "cuda:0" if torch.cuda.is_available() else "cpu" # <-- Remove or move to __init__
    # processor = OwlViTProcessor.from_pretrained(model_path) # <-- Move to __init__
    # model = OwlViTForObjectDetection.from_pretrained(model_path) # <-- Move to __init__
    # model = model.to(device) # <-- Move to __init__

    def __init__(self, device):
        print(f"Initializing Object Localization")
        self.device = device
        self.model_path = "/home/wyt/VIoTGPT/tools/owlvit"  # Or hardcode directly below

        try:
            self.processor = OwlViTProcessor.from_pretrained(self.model_path)
            self.model = OwlViTForObjectDetection.from_pretrained(self.model_path)
            # self.model = self.model.to(self.device)
            self.initialized = True
        except Exception as e:
            print(f"Failed to initialize ObjectLOCTool: {e}")
            self.initialized = False

    @prompts(
        name="Localize objects",  # Give it a clear name, consistent with other tools
        description="useful when you need to find the location (bounding box) of a specific object in an image. "
        "The tool takes an object name and an image file path and returns the bounding boxes where the object is found. "
        "The input should be a comma-separated string of two parts: the name of the object you are looking for, and the path to the image file. "
        "The input must only contain the text and the path, without any parameter names. For example, a valid input is 'a red car,./Evo_data/videos/video_79c531.mp4'.",
    )
    def inference(self, inputs: str) -> str:
        if not self.initialized:
            return "Error: Object Localization tool was not initialized properly."

        try:
            parts = inputs.split(",", 1)  # Split only on the first comma
            if len(parts) != 2:
                return "Error: Input must be in the format 'object_name,image_path'. Please provide both."

            object_query = parts[0].strip()
            image_path = parts[1].strip()

            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}. Please provide a valid image path."

            image = Image.open(image_path)
            # image = image.convert("RGB")

            texts = []
            texts.append(f"a photo of {object_query}")
            texts = [texts]

            inputs_model = self.processor(text=texts, images=image, return_tensors="pt")
            # inputs_model = inputs_model.to(self.device)
            outputs = self.model(**inputs_model)

            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs=outputs, threshold=0.1, target_sizes=target_sizes
            )

            # Format the output into a string
            i = 0
            # text = texts[i] # Not directly used in output string
            output_boxes = []

            for box, score, pred in zip(
                results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            ):
                output_boxes.append(
                    {
                        "box": [round(coord, 2) for coord in box.tolist()],
                        "score": round(score.item(), 4),
                    }
                )

            if not output_boxes:
                return f"No objects matching '{object_query}' found in {image_path}."
            else:
                box_info_strings = [
                    f"Box: {item['box']}, Score: {item['score']}"
                    for item in output_boxes
                ]
                return f"Found '{object_query}' in {image_path} at bounding box locations: {'; '.join(box_info_strings)}."

        except Exception as e:
            return f"An error occurred during object localization for input '{inputs}': {e}"


class FSDetect:
    def __init__(self, device):
        print(f"Initializing Fire and Smoke Detection")
        # 建议：尽量不要硬编码绝对路径，用 self.base_path 方便管理
        self.base_path = "/home/wyt/VIoTGPT/tools/FireSmokeDetection"
        self.out = os.path.join(self.base_path, "result/")
        self.weights = os.path.join(self.base_path, "best.pt")

        self.save_img = True
        self.imgsz = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.ratio = 0.2
        self.device = torch.device(device)

        # 1. 保存现场
        original_path = sys.path.copy()

        try:
            # 2. 把火灾检测路径强行插到第一位
            if self.base_path not in sys.path:
                sys.path.insert(0, self.base_path)
            else:
                sys.path.remove(self.base_path)
                sys.path.insert(0, self.base_path)

            # =========================================================
            # 3. [核心修改] 暴力删除内存中所有叫 'models' 或 'utils' 的缓存
            # 这一步会强迫 Python 必须去新的 sys.path[0] (即火灾目录) 重新加载代码
            # =========================================================

            modules_to_kill = [
                "models",
                "models.common",
                "models.experimental",
                "utils",
                "utils.general",
            ]

            # 找出所有相关的已加载模块
            keys_to_remove = []
            for k in sys.modules:
                for m in modules_to_kill:
                    if k == m or k.startswith(m + "."):
                        keys_to_remove.append(k)
                        break

            # 删掉它们！
            for k in keys_to_remove:
                del sys.modules[k]

            # 4. 在当前上下文重新导入 attempt_load
            # 必须在这里 import，不能在文件开头 import，否则绑定的是旧的
            from models.experimental import attempt_load

            # 5. 加载模型
            self.model = attempt_load(self.weights, map_location=self.device)
            print("  ✓ Fire and Smoke Detection model loaded successfully.")

        except Exception as e:
            print(f"  ✗ Failed to initialize FSDetect: {e}")
            # 打印错误方便调试
            import traceback

            traceback.print_exc()

        finally:
            # 6. 恢复 sys.path，以免影响后续代码查找其他库
            sys.path = original_path

    @prompts(
        name="Fire and Smoke Detection",
        description="useful when you want to know whether there is fire or smoke in the video, receives video_path as input. "
        "The input to this tool should be a string, representing the video_path. "
        "Do not include parameter names like 'video_path=' in the input. For example, a valid input is './Evo_data/videos/video_79c544.mp4'.",
    )
    def inference(self, video_path):
        try:
            if "/" in video_path:
                # source = (
                #     video_path.split("/")[0] + "/" + video_path.split("/")[-1]
                # )
                filename = os.path.basename(video_path)

                # 2. 获取所有路径组件
                #    首先规范化路径以处理 '..' 或 '.'
                norm_path = os.path.normpath(video_path)
                #    使用 os.path.sep 分隔，确保跨平台
                path_parts = norm_path.split(os.path.sep)

                # # 3. 拼接结果
                # if len(path_parts) > 1:
                #     # 使用 os.path.join 来安全地拼接，它会自动使用正确的分隔符
                #     source = os.path.join(path_parts[0], filename)
                # else:
                #     source = filename
                source = norm_path
            else:
                final_msg = "Video {} does not exists.".format(video_path)
                return final_msg

            vid_path, vid_writer = None, None
            try:
                dataset = LoadImages(source, img_size=self.imgsz)
            except Exception as e:
                msg = e
                return msg, None
            names = (
                self.model.module.names
                if hasattr(self.model, "module")
                else self.model.names
            )
            colors = [
                [random.randint(0, 255) for _ in range(3)] for _ in range(len(names))
            ]

            with torch.no_grad():
                img = torch.zeros(
                    (1, 3, self.imgsz, self.imgsz), device=self.device
                )  # init img
                _ = self.model(img)  # run once
                cla = [False, False]  # mark of fire and smoke
                queue_smoke = queue.Queue(100)
                queue_fire = queue.Queue(100)
                for _ in range(99):
                    queue_smoke.put(0)
                    queue_fire.put(0)
                smoke = 0
                fire = 0
                nframes = 0

                for path, img, im0s, vid_cap in dataset:
                    nframes += 1
                    img = torch.from_numpy(img).to(self.device)
                    img = img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = self.model(img)[0]

                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                    for i, det in enumerate(pred):
                        p, s, im0 = path, "", im0s
                        save_path = str(Path(self.out) / Path(p).name)
                        txt_path = str(Path(self.out) / Path(p).stem) + (
                            "_%g" % dataset.frame if dataset.mode == "video" else ""
                        )
                        s += "%gx%g " % img.shape[2:]
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        input_fire = False
                        input_smoke = False
                        if det is not None and len(det):
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape
                            ).round()
                            # Print results
                            for c in det[:, -1].unique():
                                if names[int(c)] == "fire":
                                    input_fire = True
                                    fire += 1
                                elif names[int(c)] == "smoke":
                                    input_smoke = True
                                    smoke += 1
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += "%g %ss, " % (n, names[int(c)])  # add to string
                            # Write results
                            for *xyxy, conf, cls in det:
                                if self.save_img or self.view_img:  # Add bbox to image
                                    label = "%s %.2f" % (names[int(cls)], conf)
                                    plot_one_box(
                                        xyxy,
                                        im0,
                                        label=label,
                                        color=colors[int(cls)],
                                        line_thickness=3,
                                    )
                        queue_fire.put(1) if input_fire else queue_fire.put(0)
                        queue_smoke.put(1) if input_smoke else queue_smoke.put(0)
                        if fire / 100 > self.ratio:
                            cla[0] = True
                        if smoke / 100 > self.ratio:
                            cla[1] = True
                        output_fire = queue_fire.get()
                        output_smoke = queue_smoke.get()
                        if output_fire == 1:
                            fire -= 1
                        if output_smoke == 1:
                            smoke -= 1
                        if self.save_img:
                            if dataset.mode == "images":
                                cv2.imwrite(save_path, im0)
                            else:
                                if vid_path != save_path:
                                    vid_path = save_path
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()

                                    fourcc = "mp4v"
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    vid_writer = cv2.VideoWriter(
                                        save_path,
                                        cv2.VideoWriter_fourcc(*fourcc),
                                        fps,
                                        (w, h),
                                    )
                                vid_writer.write(im0)
                if nframes < 100:
                    cla[0] = True if fire / nframes > self.ratio else False
                    cla[1] = True if smoke / nframes > self.ratio else False
                if cla[0] and cla[1]:
                    msg = "Dangerous! We have detected both smoke and fire in video {}.".format(
                        video_path
                    )
                elif cla[0] and not cla[1]:
                    msg = "Dangerous! We have detected fire in video {}.".format(
                        video_path
                    )
                elif not cla[0] and cla[1]:
                    msg = "Dangerous! We have detected smoke in video {}.".format(
                        video_path
                    )
                else:
                    msg = "There is no smoke or fire detected in video {}.".format(
                        video_path
                    )

            return msg
        except:
            return "Unkown Error in FSDetect."


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


class ImageSegmenter:
    def __init__(self, device="cuda"):
        print("Initializing Image Segmenter (SAM2)")

        os.makedirs("images", exist_ok=True)

        sam2_model = build_sam2(
            "configs/sam2/sam2_hiera_l.yaml",
            "/home/wyt/VIoTGPT/sam2/checkpoints/sam2_hiera_large.pt",
        ).to(device)
        sam2_model.eval()

        self.predictor = SAM2ImagePredictor(sam2_model)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

        # Utility for creating unique filenames for the output images
        alphabet = string.ascii_lowercase + string.digits
        self.su = shortuuid.ShortUUID(alphabet=alphabet)
        # print("Image Segmenter initialized.")

    @prompts(
        name="Segment Objects in an Image",
        description="Useful when you want to segment objects in an image. "
        "This tool can run in two modes. "
        "1. Segment Anything Mode: To segment all objects in an image, provide only the image path. "
        "2. Prompted Mode: To segment a specific object, provide the image path and a bounding box [x1, y1, x2, y2]. "
        "The input must be a comma-separated string. In both cases, the input must only contain the required values (path and optionally coordinates), without any parameter names like 'image_path='. "
        "For example, to segment anything, use './Evo_data/images/image_68a5b9.jpg'. "
        "To segment a specific object, use './Evo_data/images/image_68a5b9.jpg,250,400,550,600'.",
    )
    def inference(self, inputs: str) -> str:
        try:
            parts = [p.strip() for p in inputs.split(",")]
            image_path = parts[0]

            if not os.path.exists(image_path):
                return f"Error: Image not found at {image_path}"

            image_raw = Image.open(image_path).convert("RGB")
            image = np.array(image_raw)

            output_image_path = os.path.join(
                "images", f"seg_{self.su.random(length=8)}.jpg"
            )

            # Mode 1: Segment Anything (only image_path is provided)
            if len(parts) == 1:
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    masks = self.mask_generator.generate(image)

                plt.figure(figsize=(12, 12))
                plt.imshow(image)
                show_anns(masks)
                plt.axis("off")
                plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
                plt.close()
                return f"Successfully segmented the image using 'Segment Anything' mode. The result with all masks is saved as '{output_image_path}'."

            # Mode 2: Prompted Segmentation (image_path and bbox are provided)
            elif len(parts) == 5:
                bbox = np.array([[float(p) for p in parts[1:]]])

                self.predictor.set_image(image)
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox,
                    multimask_output=False,
                )

                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(masks.squeeze(0), plt.gca(), random_color=True)
                show_box(bbox[0], plt.gca())
                plt.axis("off")
                plt.savefig(output_image_path, bbox_inches="tight", pad_inches=0)
                plt.close()
                return f"Successfully segmented the object with the given bounding box. The visualization is saved as '{output_image_path}'."

            else:
                return "Invalid input format. Please provide either 'image_path' or 'image_path,x1,y1,x2,y2'."

        except Exception as e:
            return f"An unknown error occurred in ImageSegmenter: {e}"


POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)


class HumanPoseEstimation:
    def __init__(self, device):
        """
        Initializes the Human Pose Estimation tool using MMPose.
        This method is called once when the agent starts to load the models.
        """
        print(f"Initializing Human Pose Estimation (MMPose)")
        self.filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
        self.pose2d_weights = "/home/wyt/VIoTGPT/tools/mmpose/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth"
        self.pose2d_config = "/home/wyt/VIoTGPT/tools/mmpose/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py"
        for model in POSE2D_SPECIFIC_ARGS:
            if self.pose2d_config is not None and model in self.pose2d_config:
                self.filter_args.update(POSE2D_SPECIFIC_ARGS[model])
                break
        self.model = MMPoseInferencer(
            pose2d=self.pose2d_config,
            pose2d_weights=self.pose2d_weights,
            scope="mmpose",
            device=None,
            det_model=None,
            det_weights=None,
            det_cat_ids=0,
            pose3d=None,
            pose3d_weights=None,
            show_progress=False,
        )

    @prompts(
        name="Human Pose Estimation",
        description="useful for when you want to estimate and visualize the human pose in an image or video. It identifies key body points and their skeletons for every person in the image or video. The input to this tool should be a string representing the image_path or video_path. It returns a description and the filename of the visualized result."
        "Do not include parameter names like 'image_path' and 'video_path=' in the input. For example, a valid input is './Evo_data/videos/video_691797.mp4'.",
    )
    def inference(self, inputs):
        input_path = inputs.strip()

        if not os.path.exists(input_path):
            return f"Error: Input file {input_path} does not exist."

        if input_path.endswith((".mp4", ".avi", ".mov")):
            is_video = True
            output_dir = "videos"
        else:
            is_video = False
            output_dir = "images"

        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.basename(input_path)
        name, ext = os.path.splitext(file_name)
        output_filename = f"{name}_pose{ext}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            call_args = {
                "inputs": input_path,
                "vis_out_dir": output_dir,  # MMPose 会在这个目录下保存结果
                "draw_bbox": True,
                "draw_heatmap": False,
                "bbox_thr": self.filter_args["bbox_thr"],
                "nms_thr": self.filter_args["nms_thr"],
                "kpt_thr": 0.3,
                "tracking_thr": 0.3,
                "thickness": 2,
                "radius": 3,
            }

            result_generator = self.model(**call_args)

            frame_count = 0
            for result in result_generator:
                frame_count += 1

            # 预期默认生成的文件名
            expected_saved_path = os.path.join(output_dir, file_name)
            # 我们想要的目标文件名
            target_final_path = os.path.join(output_dir, f"pose_{file_name}")

            if os.path.exists(expected_saved_path):
                # 情况1：输入输出路径相同，无法重命名（否则源文件也没了）
                if expected_saved_path == input_path:
                    print("Warning: Overwriting input file, skipping rename.")
                    final_path = expected_saved_path # 保持原名
                    
                # 情况2：正常情况，进行重命名
                else:
                    # 为了安全，如果目标文件已存在，先删除，避免 Windows 下报错
                    if os.path.exists(target_final_path):
                        os.remove(target_final_path)
                        
                    os.rename(expected_saved_path, target_final_path)
                    final_path = target_final_path # 更新为新名字
            else:
                # 情况3：没找到生成的文件
                print(f"Error: Output file {expected_saved_path} not found!")
                # 这里应该直接 return 错误信息，而不是继续瞎猜
                # 但为了保持变量，暂时设回默认值，后续检查 final_path 是否存在即可
                final_path = expected_saved_path

            if is_video:
                return f"Human pose estimation completed. The video with pose visualization is saved at {final_path}."
            else:
                return f"Human pose estimation completed. The image with pose visualization is saved at {final_path}."

        except Exception as e:
            import traceback

            traceback.print_exc()
            # return f"Error running Human Pose Estimation: {str(e)}"
            return "Unknown error in Human Pose Estimation."


# class WeaponDetector:
#     """
#     一个用于检测图像中特定武器（刀、枪）的工具。
#     它使用YOLOv8模型进行目标检测。
#     ！！！已被弃用
#     """
#     def __init__(self, device):

#         print("Initializing Weapon Detector...")
#         # 将模型加载移到构造函数中，避免重复加载

#         self.model_path = './tools/Weapons-and-Knives-Detector/runs/detect/Normal_Compressed/weights/best.pt'
#         if not os.path.exists(self.model_path):
#              raise FileNotFoundError(f"Weapon detection model not found at {self.model_path}. Please check the path.")
#         self.model = YOLO(self.model_path)
#         self.conf_threshold = 0.5
#         print("Weapon Detector initialized successfully.")

#     @prompts(
#         name="Detect Knife and Gun",
#         description="useful when you want to detect knives and guns in a given image. The input to this tool should be a string, representing the path to the image."
#     )
#     def inference(self, image_path: str) -> str:
#         """
#         Performs knife and gun detection on a single image.

#         Args.
#             image_path (str): path of the image file to be tested.

#         Returns.
#             str: a string describing the detection result.
#                  If the object is found, the object's category and bounding box coordinates are returned.
#                  If not found, a message that the object was not found is returned.
#                  If the file does not exist or an error occurred, an error message is returned.
#         """
#         try:

#             if not os.path.exists(image_path):
#                 return f"Error: Image not found at path: {image_path}"


#             # YOLOv8的'source'参数可以直接接收路径，无需cv2.imread
#             results = self.model(image_path, verbose=False) # 设置verbose=False以减少控制台输出

#             detections_found = []

#             for result in results:
#                 boxes = result.boxes
#                 for i in range(len(boxes)):
#                     if boxes.conf[i] >= self.conf_threshold:

#                         class_name = result.names[int(boxes.cls[i])]

#                         box = boxes.xyxy[i].cpu().numpy().astype(int)

#                         bbox_str = f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
#                         detections_found.append(f"{class_name} at {bbox_str}")


#             if not detections_found:
#                 return f"No knife or gun was found in the image {image_path}."
#             else:
#                 # 将所有检测结果合并为一个字符串
#                 detection_summary = ', '.join(detections_found)
#                 return f"In the image {image_path}, I found {len(detections_found)} weapon(s): {detection_summary}."


#         except Exception as e:
#             return f"An unknown error occurred in WeaponDetector: {e}"
class GroundingDINOWeaponDetector:
    def __init__(self, device):
        """
        初始化通用物体检测器 (Grounding DINO)
        """
        print(f"Initializing Object Detector (Grounding DINO)")

        model_id = "/home/wyt/VIoTGPT/tools/grounding_dino"
        self.device = device
        self.box_threshold = 0.35  # 设置检测阈值

        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self.device)
            print("Object Detector loaded successfully.")
        except Exception as e:
            print(f"Error loading Grounding DINO model: {e}")
            raise

    def _visualize_and_save(self, image_path, boxes, scores, labels):
        """
        一个私有辅助方法，用于在图像上绘制检测结果并保存。
        返回新图像的路径。
        """
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            print(f"Error: OpenCV could not read image at {image_path}")
            return None, "Failed to read image for visualization."

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            color = (0, 0, 255)  # 红色 (BGR)
            thickness = 2
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{label}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            text_color = (255, 255, 255)  # 白色

            (w, h), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            cv2.rectangle(img_cv2, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(
                img_cv2,
                label_text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                text_color,
                text_thickness,
            )

        output_filename = f"det_{shortuuid.uuid()}.png"
        output_image_path = os.path.join("images", output_filename)

        cv2.imwrite(output_image_path, img_cv2)

        return output_image_path

    @prompts(
        name="Detect Knife and Gun",
        description="useful when you want to detect knives and guns in a given image. "
        "The tool can detect multiple knives and guns at once. "
        "The input to this tool should be a comma-separated string of two parts: the image_path and the text_prompt. "
        "The input must only contain the path and text, without parameter names. For example, a valid input is './Evo_data/images/image_68a5b9.jpg, a knife, a gun'.",
    )
    def inference(self, inputs):
        """
        Performs object detection inference.
        :param inputs: a string containing "image_path,text_prompt"
        :return: a string describing the detection result.
        """
        try:
            parts = [p.strip() for p in inputs.split(",", 1)]
            if len(parts) != 2:
                return "Error: Invalid input format. Please provide the image path and text prompt separated by a comma."

            image_path, text_prompt = parts

            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            image_pil = Image.open(image_path).convert("RGB")
            text_prompt = "a knife . a blade . a weapon in hand ."

            processed_inputs = self.processor(
                images=image_pil, text=text_prompt, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**processed_inputs)

            target_sizes = torch.tensor([image_pil.size[::-1]])
            results = self.processor.post_process_grounded_object_detection(
                outputs, processed_inputs.input_ids, target_sizes=target_sizes
            )[0]  # 我只处理一张图片

            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]

            mask = scores > self.box_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [label for i, label in enumerate(labels) if mask[i]]

            if len(boxes) == 0:
                return f"No objects matching the description '{text_prompt}' were found in the image {image_path} with a confidence above {self.box_threshold}."

            output_image_path = self._visualize_and_save(
                image_path, boxes, scores, labels
            )

            descriptions = []
            for label, score, box in zip(labels, scores, boxes):
                descriptions.append(f"a '{label}' with confidence {score:.2f}")

            descriptions_str = ", ".join(descriptions)
            final_msg = (
                f"Successfully detected {len(boxes)} objects: {descriptions_str}. "
                f"The visualized image is saved as {output_image_path}."
            )

            print("final_msg: ", final_msg)
            return final_msg

        except Exception as e:
            print(f"An error occurred in ObjectDetector: {e}")
            return "Unknown error in ObjectDetector."


class SceneRecognition:
    def __init__(self, device):
        print("Initializing Scene Recognition")
        self.device = device
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = (
            self.__load_labels()
        )
        self.features_blobs = []
        self.model = self.__load_model().to(self.device)
        params = list(self.model.parameters())
        self.weight_softmax = params[-2].data
        self.weight_softmax[self.weight_softmax < 0] = 0
        self.ratio = 0.1
        self.trasform = pd.read_csv(
            "/home/wyt/VIoTGPT/tools/place/transform.txt", header=None, index_col=0
        ).to_dict()[1]

    def __recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.__recursion_change_bn(module1)
        return module

    def __load_labels(self):
        file_name_category = "/home/wyt/VIoTGPT/tools/place/categories_places365.txt"
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(" ")[0][3:])
        classes = tuple(classes)
        # indoor and outdoor relevant
        file_name_IO = "/home/wyt/VIoTGPT/tools/place/IO_places365.txt"
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)
        labels_IO = np.array(labels_IO)
        # scene attribute relevant
        file_name_attribute = "/home/wyt/VIoTGPT/tools/place/labels_sunattribute.txt"
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = "/home/wyt/VIoTGPT/tools/place/W_sceneattribute_wideresnet18.npy"
        W_attribute = np.load(file_name_W)
        return classes, labels_IO, labels_attribute, W_attribute

    def __returnTF(self):
        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return tf

    def __load_model(self):
        # this model has a last conv feature map as 14x14
        model_file = "/home/wyt/VIoTGPT/tools/place/wideresnet18_places365.pth.tar"
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
        for i, (name, module) in enumerate(model._modules.items()):
            module = self.__recursion_change_bn(model)
        model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        model.eval()
        return model

    @prompts(
        name="Recognize the Scene in the Video",
        description="useful when you want to determine the general semantic class of the scene in the video. "
        "The input to this tool should be a string, representing the video_path. ",
    )
    def inference(self, inputs):
        video_path = inputs
        if not os.path.exists(video_path):
            return f"Error: Video file not found at '{video_path}'"
        # video_path = video_path.split('/')[0] + '/Anomaly_' + video_path.split('/')[-1]
        video_cap = cv2.VideoCapture(video_path)
        nframes = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, img = video_cap.read()
        scene_idx_count = [0 for i in range(365)]
        if nframes > 50000:
            return "other"
        for i in tqdm(range(nframes)):
            if not ret:
                break
            img = Image.fromarray(img)
            input_img = self.__returnTF()(img).unsqueeze(0).to(self.device)

            # forward pass
            with torch.no_grad():
                logit = self.model.forward(input_img)
                h_x = Fun.softmax(logit, 1).data.squeeze()
                idx = torch.max(h_x, 0)[1]
                scene_idx_count[idx] += 1
            ret, img = video_cap.read()
        scene = self.classes[scene_idx_count.index(max(scene_idx_count))]
        if scene in self.trasform.keys():
            scene = self.trasform[scene]
        else:
            scene = "an unknown place"
        return scene

class VideoAnomalyDetection:
    def __init__(self, device):
        print(f"Initializing Video Anomaly Detection (API Client)")
        # 不再在主进程加载模型，而是配置本地 API 的地址
        # 支持通过环境变量 HOLMES_PORT 配置端口，默认 8000
        port = os.environ.get("HOLMES_PORT", "8000")
        self.api_url = f"http://127.0.0.1:{port}/detect"

    @prompts(
        name="Detect and Analyze Anomalies in Video",
        description="Useful when you want to describe video activities or find unusual/anomalous events in a video. "
        "The tool will analyze the video and return a text description of any detected anomalies. "
        "The input to this tool should be the scene class of the video and the path to the video file"
        "Do not include parameter names like 'video_path=' in the input. For example, a valid input is 'an unknown place,./Evo_data/videos/video_678457.mp4'.",
    )
    def inference(self, inputs):
        """
        Analyzes a video to detect and describe anomalous human activities via HTTP Request.
        """
        try:
            # 发起 POST 请求调用我们在后台运行的模型服务
            response = requests.post(self.api_url, json={"inputs": inputs})
            response.raise_for_status()
            
            # 提取服务器返回的推理结果
            final_msg = response.json().get("result", "Error: No result returned from server.")
            print("VideoAnomalyDetection API Result: ", final_msg)
            return final_msg

        except requests.exceptions.ConnectionError:
            error_msg = "Error: VideoAnomalyDetection service is offline. Please start 'holmes_server.py'."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unknown Error in API Call: {str(e)}"
            print(error_msg)
            return error_msg
# class VideoAnomalyDetection:
#     def __init__(self, device):
#         print(f"Initializing Video Anomaly Detection")
#         # 将模型加载和设置部分移到构造函数中，这样模型只会被加载一次
#         mllm_path = "/home/wyt/VIoTGPT/tools/HolmesVAU/ckpt"
#         sampler_path = (
#             "/home/wyt/VIoTGPT/tools/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth"
#         )
#         self.device = device

#         try:
#             (self.model, self.tokenizer, self.generation_config, self.sampler) = (
#                 load_model(mllm_path, sampler_path, self.device)
#             )
#             print("Video Anomaly Detection model loaded successfully.")
#         except Exception as e:
#             print(f"Error loading Video Anomaly Detection model: {e}")

#             # 报错时抛出异常，将模型设置为 None
#             self.model = None

#     @prompts(
#         name="Detect and Analyze Anomalies in Video",
#         description="Useful when you want to describe video activities or find unusual/anomalous events in a video. "
#         "The tool will analyze the video and return a text description of any detected anomalies. "
#         "The input to this tool should be the scene class of the video and the path to the video file"
#         "Do not include parameter names like 'video_path=' in the input. For example, a valid input is 'an unknown place,./Evo_data/videos/video_678457.mp4'.",
#     )
#     def inference(self, inputs):
#         """
#         Analyzes a video to detect and describe anomalous human activities.
#         Args:
#             inputs (str): Comma-separated string: `<scene_class>, <video_path>`
#         Returns:
#             str: A string describing the detected anomalies or an error message.
#         """

#         if self.model is None:
#             return "Human Activity Analysis tool is not available due to a model loading error."
        
        
        

#         try:
#             if "," in inputs:
#                 scene_class, video_path = [part.strip() for part in inputs.split(",", 1)]
#             else:
#                 scene_class = "noscene"
#                 video_path = inputs.strip()

#             if not os.path.exists(video_path):
#                 return f"Error: Video file not found at '{video_path}'"

#             # 提示语可以固定，因为这个工具的目的很明确
#             prompt = "Could you specify the anomaly events present in the video?"

#             pred, history, frame_indices, anomaly_score = generate(
#                 video_path=video_path,
#                 prompt=prompt,
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 generation_config=self.generation_config,
#                 sampler=self.sampler,
#                 select_frames=12,
#                 use_ATS=True,
#             )

#             # 格式化输出，使其对智能体更友好
#             if pred and scene_class != "noscene":
#                 final_msg = f"The scene of the video is {scene_class}. And the analysis of video is {pred}."
#             elif pred and scene_class == "noscene":
#                 final_msg = f"The analysis of video is {pred}."
#             elif not pred and scene_class != "noscene":
#                 final_msg = f"The scene of the video is {scene_class}. No specific anomalies detected in video '{video_path}'."
#             else:                 
#                 final_msg = f"No specific anomalies detected in video '{video_path}'."

#             print("final_msg: ", final_msg)
#             return final_msg

#         except Exception as e:
#             print(f"An error occurred during video analysis: {e}")
#             return "Unknown Error in HumanActivityAnalysis."


def get_image_path(folder="images"):
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"crowd_counting_result_{shortuuid.uuid()}.png")


class CrowdCounting:
    def __init__(self, device="cuda"):
        print("Initializing Crowd Counting Tool")
        self.device = device

        args_config = argparse.Namespace(
            dataset="jhu",
            save_path="save_file/A_ddp",
            workers=2,
            print_freq=200,
            start_epoch=0,
            epochs=5000,
            pre=None,
            batch_size=16,
            crop_size=256,
            lr_step=1200,
            seed=1,
            best_pred=100000.0,
            gpu_id="0,1",
            lr=0.0001,
            weight_decay=0.0005,
            save=False,
            scale_aug=False,
            scale_type=0,
            scale_p=0.3,
            gray_aug=False,
            gray_p=0.1,
            test_patch=False,
            channel_point=3,
            num_patch=1,
            min_num=-1,
            num_knn=4,
            test_per_epoch=20,
            threshold=0.35,
            video_path="./video_demo/1.mp4",
            local_rank=-1,
            lr_backbone=0.0001,
            lr_drop=40,
            clip_max_norm=0.1,
            frozen_weights=None,
            backbone="resnet50",
            dilation=False,  # 注意：原始代码可能是True，根据模型需要调整
            position_embedding="sine",
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=700,  # 确保这个值和模型训练时一致(训练时700)
            pre_norm=False,
            masks=False,
            aux_loss=True,
            set_cost_class=2,
            set_cost_point=5,
            set_cost_giou=2,
            mask_loss_coef=1,
            dice_loss_coef=1,
            cls_loss_coef=2,
            count_loss_coef=2,
            point_loss_coef=5,
            giou_loss_coef=2,
            focal_alpha=0.25,
            dataset_file="crowd_data",
            coco_path=None,
            coco_panoptic_path=None,
            remove_difficult=False,
            output_dir="",
            device=device,
            resume="",
            eval=False,
            num_workers=2,
            world_size=1,
            dist_url="env:// ",
            master_port=29501,
            distributed=False,
        )

        self.args = {
            "pre": "/home/wyt/VIoTGPT/tools/CLTR_crowdcounting/ckpt/video_model.pth",
            "num_queries": 700,
            "seed": 42,
        }

        setup_seed(self.args["seed"])
        params = {
            "pre": "/home/wyt/VIoTGPT/tools/CLTR_crowdcounting/ckpt/video_model.pth",  # 模型权重路径
            "num_queries": 700,  # 这应与模型的配置相匹配
        }

        t_args = args_config
        for k, v in params.items():
            setattr(t_args, k, v)
        self.args = t_args

        # 2. 设置随机种子和分布式模式（对于推理是可选的，但为了与原始代码保持一致）
        setup_seed(self.args.seed)
        utils.init_distributed_mode(self.args)

        # 3. 构建模型
        model, _, _ = build_model(args_config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()  # 必须设置为评估模式

        # 4. 加载预训练权重
        checkpoint_path = self.args.pre
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 权重键的名称调整逻辑（特定于此模型）
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("bbox", "point")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(
                f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})"
            )
        else:
            raise FileNotFoundError(
                f"=> CRITICAL: No checkpoint found at '{checkpoint_path}'"
            )

        self.model = model

        # 5. 定义图像转换
        self.img_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.tensor_transform = transforms.ToTensor()

        # warnings.filterwarnings('ignore')

    def _show_map(self, out_pointes, frame, width, height, crop_size, num_h, num_w):
        """
        内部辅助方法，用于从模型输出重建点图和密度图，并进行可视化。
        （逻辑与原始脚本中的 show_map 函数相同）
        """
        kpoint_list = []
        for i in range(len(out_pointes)):
            out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
            out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
            k = np.zeros((crop_size, crop_size))

            for j in range(len(out_point)):
                if out_value[j] < 0.25:  # 置信度阈值
                    break
                y = min(int(out_point[j][0]), crop_size - 1)
                x = min(int(out_point[j][1]), crop_size - 1)
                k[x, y] = 1
            kpoint_list.append(k)

        kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
        kpoint = (
            kpoint.view(num_h, num_w, crop_size, crop_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(num_h * crop_size, num_w * crop_size)
            .cpu()
            .numpy()
        )

        density_map = gaussian_filter(kpoint.copy(), 6)
        density_map = density_map / (np.max(density_map) + 1e-8) * 255
        density_map = density_map.astype(np.uint8)
        density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

        pred_coor = np.nonzero(kpoint)
        count = len(pred_coor[0])

        point_map = (
            np.zeros((int(kpoint.shape[0]), int(kpoint.shape[1]), 3), dtype="uint8")
            + 255
        )
        for i in range(count):
            w = int(pred_coor[1][i])
            h = int(pred_coor[0][i])
            cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
            cv2.circle(frame, (w, h), 3, (0, 255, 50), -1)

        return point_map, density_map, frame, count

    @prompts(
        name="Estimate the Crowd Count in an Image",
        description="Useful for when you need to count the number of people in a given image. "
        "The tool will process the image, detect individuals, and return the total count. "
        "The input to this tool should be a single string representing the image_path. "
        "Do not include parameter names like 'video_path=' in the input. For example, a valid input is './Evo_data/videos/video_643432.mp4'.",
    )
    def inference(self, inputs):
        image_path = inputs.strip()

        try:
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            frame = cv2.imread(image_path)
            if frame is None:
                return f"Error: Could not read image from path: {image_path}"

            # --- 图像预处理 ---
            # 调整大小以适应模型的切片逻辑（必须是256的倍数）
            width, height = 1024, 768
            frame = cv2.resize(frame, (width, height))
            ori_frame = frame.copy()

            image = self.tensor_transform(frame)
            image = self.img_transform(image)

            # 图像切片逻辑
            img_w, img_h = image.shape[2], image.shape[1]
            num_w = int(img_w / 256)
            num_h = int(img_h / 256)
            image = image.view(3, num_h, 256, img_w).view(3, num_h, 256, num_w, 256)
            image = (
                image.permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(3, num_w * num_h, 256, 256)
                .permute(1, 0, 2, 3)
            )

            # --- 模型推理 ---
            with torch.no_grad():
                image = image.to(self.device)
                outputs = self.model(image)

                out_logits, out_point = outputs["pred_logits"], outputs["pred_points"]
                prob = out_logits.sigmoid()
                topk_values, topk_indexes = torch.topk(
                    prob.view(out_logits.shape[0], -1), self.args.num_queries, dim=1
                )
                topk_points = topk_indexes // out_logits.shape[2]
                out_point = torch.gather(
                    out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2)
                )
                out_point = out_point * 256  # 将坐标从[0,1]缩放到[0, 256]
                value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)

            # --- 后处理和可视化 ---
            crop_size = 256
            kpoint_map, density_map, annotated_frame, count = self._show_map(
                value_points, frame, img_w, img_h, crop_size, num_h, num_w
            )

            # 将4个视图合并为一张结果图
            res1 = np.hstack((ori_frame, kpoint_map))
            res2 = np.hstack((density_map, annotated_frame))
            res = np.vstack((res1, res2))
            cv2.putText(
                res,
                f"Count: {count}",
                (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                5,
            )

            # 保存结果图并返回信息
            output_path = get_image_path()
            cv2.imwrite(output_path, res)

            final_msg = f"Successfully counted {count} people in the image '{image_path}'. A visualization has been saved to '{output_path}'."
            print(final_msg)
            return final_msg

        except Exception as e:
            return f"An error occurred during crowd counting: {e}"


def yaml_to_dict(path: str):
    """Reads a yaml file into a dict."""
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def get_color(obj_id, rgb=False, use_int=True):
    """
    Generates a consistent color for a given object ID.
    This is a simple implementation to replace the notebook's `get_color`.
    """
    np.random.seed(obj_id)
    color = np.random.randint(0, 255, 3)
    if use_int:
        color = tuple(int(c) for c in color)
    if not rgb:
        color = (color[2], color[1], color[0])  # BGR format for OpenCV
    return color


class HumanPoseTracking:
    def __init__(self, device="cuda"):
        print(f"Initializing Human Pose and Trajectory Tracking Tool.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This tool requires a CUDA-enabled GPU."
            )

        self.device = device
        self.dtype = torch.float16

        config_path = "/home/wyt/VIoTGPT/tools/MOTIP_MultipleObjectTracking/configs/r50_deformable_detr_motip_sportsmot.yaml"
        checkpoint_path = "/home/wyt/VIoTGPT/tools/MOTIP_MultipleObjectTracking/outputs/r50_deformable_detr_motip_sportsmot/r50_deformable_detr_motip_sportsmot.pth"

        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model config or checkpoint not found. "
                f"Please ensure '{config_path}' and '{checkpoint_path}' exist."
            )

        config = yaml_to_dict(config_path)
        config = load_super_config(config, config["SUPER_CONFIG_PATH"])

        model, _ = build_TrackingModel(config)
        load_checkpoint(model, checkpoint_path)
        model.eval()
        model = model.to(self.device)
        if self.dtype == torch.float16:
            model.half()

        self.model = model
        print("Model built and loaded successfully.")

    def _simple_transform(self, image, max_shorter=800, max_longer=1440):
        """Preprocesses a single image frame."""
        image_tensor = F.to_tensor(image)
        # Note: F.resize expects (C, H, W)
        h, w = image_tensor.shape[1:]
        scale = max_shorter / min(h, w)
        if h < w:
            new_h, new_w = max_shorter, int(w * scale)
        else:
            new_h, new_w = int(h * scale), max_shorter

        if max(new_h, new_w) > max_longer:
            scale = max_longer / max(new_h, new_w)
            new_h, new_w = int(new_h * scale), int(new_w * scale)

        resized_image = F.resize(image_tensor, size=[new_h, new_w])

        image_tensor = F.normalize(
            resized_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if self.dtype != torch.float32:
            image_tensor = image_tensor.to(self.dtype)
        return image_tensor.to(self.device)

    @prompts(
        name="Human Pose and Trajectory Tracking",
        description="Useful for when you want to track all people in a video and visualize their trajectories. "
        "This tool processes an entire video, draws bounding boxes and tracking IDs on each person, "
        "and saves the result as a new video file. "
        "The input to this tool should be a string representing the path to the video file. "
        "Do not include parameter names like 'video_path=' in the input. For example, a valid input is './Evo_data/videos/video_674333.mp4'.",
    )
    def inference(self, video_path: str):
        try:
            video_path = video_path.strip()
            if not os.path.exists(video_path):
                return f"Error: Video file not found at '{video_path}'."

            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened():
                return f"Error: Failed to open video file: {video_path}"

            fps = video_cap.get(cv2.CAP_PROP_FPS)
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_dir = "videos"
            base_name = os.path.basename(video_path)
            file_name, file_ext = os.path.splitext(base_name)
            output_path = os.path.join(
                output_dir, f"{file_name}_pose_tracked{file_ext}"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            print(f"Processing video: {video_path}. It has {length} frames.")

            runtime_tracker = RuntimeTracker(
                model=self.model,
                sequence_hw=(height, width),
                assignment_protocol="object-max",
                miss_tolerance=30,
                det_thresh=0.5,
                newborn_thresh=0.5,
                id_thresh=0.2,
                dtype=self.dtype,
            )

            for _ in tqdm(range(length), desc="Tracking Human Poses", unit="frame"):
                ret, frame = video_cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_tensor = self._simple_transform(frame_rgb)
                frame_tensor = nested_tensor_from_tensor_list([frame_tensor])

                runtime_tracker.update(frame_tensor)

                with torch.no_grad():
                    track_results = runtime_tracker.get_track_results()
                    num_persons = len(track_results["bbox"])
                for bbox, obj_id in zip(track_results["bbox"], track_results["id"]):
                    x, y, w, h = map(int, bbox)
                    color = get_color(obj_id)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        f"ID: {obj_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                video_writer.write(frame)

            video_cap.release()
            video_writer.release()

            # final_msg = f"Successfully tracked human poses in '{video_path}'. The result video is saved as '{output_path}'."
            final_msg = f"Successfully tracked human poses and find {num_persons} people in '{video_path}'. The result video is saved as '{output_path}'."
            print(final_msg)
            return final_msg

        except Exception as e:
            return f"An error occurred during human pose tracking: {e}"

class FaceRecognition:
    def __init__(self, device):
        print(f"Initializing FaceRecognition")
        self.detector = cv2.FaceDetectorYN.create(
            '/home/wyt/VIoTGPT/tools/Face_Recognition/face_detection_yunet_2023mar.onnx',  # YuNet
            "",
            (320, 320),
            0.9,  # Filtering out faces of score < score_threshold
            0.3,  # Suppress bounding boxes of iou >= nms_threshold
            5000  # Keep top_k bounding boxes before NMS
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            '/home/wyt/VIoTGPT/tools/Face_Recognition/face_recognition_sface_2021dec.onnx', "")
        self.cosine_similarity_threshold = 0.363
        self.l2_similarity_threshold = 1.128

    @prompts(name="Recognize the Face",
             description="useful when you want to know whether the faces in the photo appeared in the video. "
                         "The tool recognize people by face. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the video_path. "
                         "The input must only contain the file paths, without parameter names like 'image_path=' or 'video_path='. For example, a valid input is './Evo_data/images/image_68a5b9.jpg,./Evo_data/videos/video_79c531.mp4'.")
    def inference(self, inputs):
        try:
            # image_path, video_path = inputs.split(',')
            image_path, video_path = [part.strip() for part in inputs.split(",", 1)]
            # Detection of the uploaded image
            if image_path is not None:
                img1 = cv2.imread(cv2.samples.findFile(image_path))
            img1Width = int(img1.shape[1])
            img1Height = int(img1.shape[0])
            img1 = cv2.resize(img1, (img1Width, img1Height))
            self.detector.setInputSize((img1Width, img1Height))
            faces1 = self.detector.detect(img1)
            if faces1[1] is None:
                final_msg = 'Cannot find a face in {}'.format(image_path)
                return final_msg
            face1_align = self.recognizer.alignCrop(img1, faces1[1][0])
            face1_feature = self.recognizer.feature(face1_align)
            # video_path = video_path.split('/')[0] + '/FaceRecognition_' + video_path.split('/')[-1]
            video = cv2.VideoCapture(video_path)
            frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.detector.setInputSize([frameWidth, frameHeight])
            frame_count = 0
            return_frame = []
            return_frame_id = []
            if video.isOpened():
                while True:
                    ret, frame = video.read()
                    if frame is None:
                        break
                    if ret == True:
                        frame_count = frame_count + 1
                        frame = cv2.resize(frame, (frameWidth, frameHeight))
                        faces2 = self.detector.detect(frame)
                        if faces2[1] is not None:
                            face2_align = self.recognizer.alignCrop(frame, faces2[1][0])
                            face2_feature = self.recognizer.feature(face2_align)
                            cosine_score = self.recognizer.match(face1_feature, face2_feature,
                                                                 cv2.FaceRecognizerSF_FR_COSINE)
                            if cosine_score >= self.cosine_similarity_threshold:
                                msg = 'the same identity'
                                return_msg = 'Frame {} have {}. Cosine Similarity: {}, threshold: {}.'. \
                                    format(frame_count, msg, cosine_score, self.cosine_similarity_threshold)
                                frame_result = visualize(frame, faces2, return_msg)
                                # cv2.imwrite('image/video_{}.jpg'.format(frame_count), frame_result)
                                return_frame.append(frame_result)
                                return_frame_id.append(frame_count)
            else:
                final_msg = 'Video {} does not exists.'.format(video_path)
                return final_msg
            video.release()
            if len(return_frame_id) > 0 and len(return_frame_id) < 30:
                final_msg = 'Video {} have {}. The person appeared in {} frames, including Frame {}.'. \
                    format(video_path, msg, len(return_frame_id), return_frame_id)
            elif len(return_frame_id) >= 30:
                final_msg = 'Video {} have {}. The person appeared in {} frames, including Frame {}, etc.'. \
                    format(video_path, msg, len(return_frame_id), return_frame_id[:30])
            else:
                final_msg = 'Video {} does not have {}.'.format(video_path, 'this identity')
            return final_msg
        except:
            return "Unknown error in FaceRecognition."
class FaceDetection:
    def __init__(self, device):
        print(f"Initializing Face Detection (DSFD)")
        self.model_path = (
            "/home/wyt/VIoTGPT/tools/FaceDetection_DSFD/WIDERFace_DSFD_RES152.pth"
        )
        self.save_folder = "/home/wyt/VIoTGPT/images/"
        self.visual_threshold = 0.5
        self.device = device
        self.cuda = self.device != "cpu"

        self.cfg = widerface_640
        num_classes = len(WIDERFace_CLASSES) + 1  # +1 background
        self.net = build_ssd("test", self.cfg["min_dim"], num_classes)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        print("Finished loading DSFD model!")

        self.transform = TestBaseTransform((104, 117, 123))
        os.makedirs(self.save_folder, exist_ok=True)

    def _infer(self, img, shrink):
        if shrink != 1:
            img = cv2.resize(
                img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR
            )
        with torch.no_grad():
            x = torch.from_numpy(self.transform(img)[0]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(self.device)

        y = self.net(x)
        detections = y.data
        scale = torch.Tensor(
            [
                img.shape[1] / shrink,
                img.shape[0] / shrink,
                img.shape[1] / shrink,
                img.shape[0] / shrink,
            ]
        )
        det = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= self.cfg["conf_thresh"]:
                score = detections[0, i, j, 0].cpu().numpy()
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                det.append([pt[0], pt[1], pt[2], pt[3], score])
                j += 1
        if (len(det)) == 0:
            return np.array([[0, 0, 0, 0, 0.0]])

        det = np.array(det)
        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        return det

    def _infer_flip(self, img, shrink):
        img_flipped = cv2.flip(img, 1)
        det = self._infer(img_flipped, shrink)
        if det.ndim == 1:
            det = det.reshape(1, -1)
        det_t = np.zeros_like(det)
        det_t[:, 0] = img.shape[1] - det[:, 2]
        det_t[:, 1] = det[:, 1]
        det_t[:, 2] = img.shape[1] - det[:, 0]
        det_t[:, 3] = det[:, 3]
        det_t[:, 4] = det[:, 4]
        return det_t

    def _vis_detections(self, im, dets, output_path):
        """Draw detected bounding boxes and save the image."""
        inds = np.where(dets[:, -1] >= self.visual_threshold)[0]
        if len(inds) == 0:
            # Save the original image if no detections meet the threshold
            cv2.imwrite(output_path, im)
            return

        im_copy = im.copy()

        for i in inds:
            bbox = dets[i, :4]
            # score = dets[i, -1]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(output_path, im_copy)

    @prompts(
        name="Detect Faces in an Image",
        description="useful for detecting all human faces in a given image and providing their locations. "
        "The input to this tool should be a string representing the image_path. "
        "Do not include parameter names like 'image_path=' in the input. For example, a valid input is './Evo_data/images/image_686jhw.png'.",
    )
    def inference(self, inputs):
        image_path = str(inputs)
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                # 即使 cv2.imread 不抛出异常，它也可能在控制台打印了 libpng error
                # 我们在这里返回一个更明确的信息
                return (
                    f"Error: OpenCV (cv2.imread) failed to read the image and returned None. "
                    f"The file '{image_path}' is likely corrupted or not a valid image format. "
                    f"Please check the console logs for low-level errors like 'libpng error'."
                )
            if img is None:
                return f"Error: Could not read image from {image_path}"

            max_im_shrink = ((2000.0 * 2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
            shrink = max_im_shrink if max_im_shrink < 1 else 1

            det0 = self._infer(img, shrink)
            det1 = self._infer_flip(img, shrink)

            st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
            det_s = self._infer(img, st)

            factor = 2
            bt = (
                min(factor, max_im_shrink)
                if max_im_shrink > 1
                else (st + max_im_shrink) / 2
            )
            det_b = self._infer(img, bt)

            if max_im_shrink > factor:
                bt *= factor
                while bt < max_im_shrink:
                    det_b = np.row_stack((det_b, self._infer(img, bt)))
                    bt *= factor
                det_b = np.row_stack((det_b, self._infer(img, max_im_shrink)))

            det = np.row_stack((det0, det1, det_s, det_b))
            det = bbox_vote(det)

            inds = np.where(det[:, -1] >= self.visual_threshold)[0]
            num_faces = len(inds)

            image_id = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"face_detection_{image_id}_{shortuuid.uuid()}.jpg"
            output_path = os.path.join(self.save_folder, output_filename)

            self._vis_detections(img, det, output_path)

            if num_faces > 0:
                final_msg = f"Successfully detected {num_faces} faces in the image. The result with bounding boxes is saved as '{output_path}'."
            else:
                final_msg = f"No faces were detected in the image. The original image is saved as '{output_path}'."

            print("final_msg: ", final_msg)
            return final_msg

        except Exception as e:
            # return f"Unknown Error in FaceDetection: {e}"

            # 使用 traceback.format_exc() 来捕获完整的错误堆栈
            detailed_error_info = traceback.format_exc()

            # 打印到控制台，方便实时调试
            print("=" * 20 + " DETAILED ERROR IN FaceDetection " + "=" * 20)
            print(detailed_error_info)
            print("=" * 60)

            # 将详细信息返回给 LangChain Agent
            return (
                f"An exception occurred in FaceDetection.\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {e}\n"
                f"Full Traceback:\n{detailed_error_info}"
            )


class LowLightEnhancer:
    def __init__(self, device):
        print("Initializing Low-Light Image Enhancer")
        self.device = torch.device(device)

        self.DCE_net = LowLightEnhancer_model.enhance_net_nopool().to(self.device)
        self.DCE_net.load_state_dict(
            torch.load(
                "/home/wyt/VIoTGPT/tools/ZeroDCE_LowLightEnhancement/ZeroDCE_code/snapshots/Epoch99.pth",
                map_location=self.device,
            )
        )
        self.DCE_net.eval()
        print("Finished loading Low-Light Image Enhancer!")

    @prompts(
        name="Enhance Low-Light Image",
        description="useful for enhancing an image that is too dark. The input to this tool should be a single string, representing the image_path. "
        "Do not include parameter names like 'image_path=' in the input. For example, a valid input is './Evo_data/images/image_68a5b9.jpg'.",
    )
    def inference(self, inputs):
        image_path = inputs.strip()
        try:
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"

            with torch.no_grad():
                data_lowlight = Image.open(image_path)
                data_lowlight = np.asarray(data_lowlight) / 255.0
                data_lowlight = torch.from_numpy(data_lowlight).float()
                data_lowlight = data_lowlight.permute(2, 0, 1)
                data_lowlight = data_lowlight.to(self.device).unsqueeze(0)

                start_time = time.time()
                _, enhanced_image, _ = self.DCE_net(data_lowlight)
                end_time = time.time()
                print(
                    f"Low-light enhancement took {end_time - start_time:.4f} seconds."
                )

                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)

                result_path = os.path.join("images", f"{name}_enhanced{ext}")

                torchvision.utils.save_image(enhanced_image, result_path)

                return f"Successfully enhanced the image. The result is saved as {result_path}"
        except Exception as e:
            return f"Unknown error in LowLightEnhancer: {e}"


# class ImageSuperResolution:
#     def __init__(self, device):
#         print(f"Initializing Image Super-Resolution Tool")

#         model_path = "/home/wyt/VIoTGPT/tools/Real_ESRGAN/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth"

#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

#         self.editor = MMagicInferencer(
#             "real_esrgan", model_ckpt=model_path, device=device
#         )
#         print("Image Super-Resolution Tool initialized successfully.")

#     @prompts(
#         name="Image Super Resolution",
#         description="useful when you want to enhance the resolution of a low-quality image, making it clearer and more detailed. The input to this tool should be the path to the image you want to enhance. Do not include parameter names like 'image_path=' in the input. For example, a valid input is './Evo_data/images/image_68a511.png'.",
#     )
#     def inference(self, inputs):
#         """
#         Enhances the resolution of an image using Real-ESRGAN.
#         Args:
#             inputs (str): The file path of the image to be enhanced.
#         Returns:
#             str: A message indicating the result, including the path to the saved high-resolution image.
#         """
#         try:
#             image_path = inputs.strip()

#             if not os.path.exists(image_path):
#                 return f"Error: The input image path '{image_path}' does not exist."
#             output_dir = "images"
#             os.makedirs(output_dir, exist_ok=True)
#             # 生成一个随机文件名以避免冲突
#             output_filename = f"super_resolution_{shortuuid.uuid()}.png"
#             result_out_path = os.path.join(output_dir, output_filename)

#             print(f"Enhancing image: {image_path}...")

#             self.editor.infer(img=image_path, result_out_dir=result_out_path)

#             final_msg = f"Successfully enhanced the image. The high-resolution image is saved at {result_out_path}"
#             print(final_msg)
#             return final_msg

#         except Exception as e:
#             print(f"An error occurred: {e}")
#             return "Unknown Error in ImageSuperResolution tool."


class HumanFallDetection:
    def __init__(self, device):
        """
        Initializes the Fall Detector model and configuration.
        This is called once when the agent is set up.
        """
        print("Initializing Human Fall Detection")
        # Initialize the detector with parameters from the config file.
        self.detector = FallDetectorMulti(
            fps=FPS,
            window_size=WINDOW_SIZE,
            v_thresh=V_THRESH,
            dy_thresh=DY_THRESH,
            ar_thresh=ASPECT_RATIO_THRESH,
        )
        # Define the directory to save processed videos.
        self.output_dir = "videos"
        os.makedirs(self.output_dir, exist_ok=True)

    @prompts(
        name="Detect Human Fall",
        description="useful when you want to detect if a person falls down in a video. "
        "This tool analyzes the entire video, detects human poses, and identifies fall events. "
        "The input to this tool should be a string, representing the video_path. "
        "Do not include parameter names like 'video_path=' in the input. For example, a valid input is './Evo_data/videos/video_6jifer.mp4'.",
    )
    def inference(self, inputs: str):
        """
        Processes a single video file to detect falls.

        Args:
            inputs (str): The path to the video file to be processed.

        Returns:
            str: A message indicating whether a fall was detected, the timestamp,
                 and the path to the output video with visualizations.
        """
        try:
            video_path = inputs.strip()

            # Check if the video file exists
            if not os.path.exists(video_path):
                return f"Error: Video file not found at '{video_path}'"

            print(f"Processing video for fall detection: {video_path}")

            # The process_video_file method should be designed to return information about falls.
            # Here, we assume it's modified to return a list of timestamps (in seconds) where falls occurred.
            # It also saves the processed video to the output directory.
            # We pass the full output path to the function.
            video_filename = os.path.basename(video_path)
            output_video_path = os.path.join(self.output_dir, video_filename)

            # Let's assume process_video_file is modified to return fall events
            fall_events = self.detector.process_video_file(
                video_path, output_video_path
            )

            if fall_events and len(fall_events) > 0:
                # Format the timestamps for a clean output message
                timestamps_str = ", ".join([f"{t:.2f}s" for t in fall_events])
                final_msg = (
                    f"A fall was detected in the video '{video_path}'. "
                    f"The event(s) occurred at approximately: {timestamps_str}. "
                    f"A video with visualizations has been saved to '{output_video_path}'."
                )
            else:
                final_msg = f"No fall was detected in the video '{video_path}'."

            print(f"Final message: {final_msg}")
            return final_msg

        except Exception as e:
            return f"An error occurred during fall detection: {e}"


class ViolenceDetection:
    def __init__(self, device):
        print("Initializing VadCLIP Violence Detection")
        self.device = device
        self.model_path = "/home/wyt/VIoTGPT/tools/VadCLIP/models/model_ucf.pth"
        self.embed_dim = 512
        self.visual_length = 256
        self.visual_width = 512
        self.visual_head = 1
        self.visual_layers = 2
        self.attn_window = 8
        self.prompt_prefix = 10
        self.prompt_postfix = 10
        self.classes_num = 14
        self.sample_rate = 16

        try:
            self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device)
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise

        self.model = CLIPVAD(
            self.classes_num,
            self.embed_dim,
            self.visual_length,
            self.visual_width,
            self.visual_head,
            self.visual_layers,
            self.attn_window,
            self.prompt_prefix,
            self.prompt_postfix,
            device,
        )

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")

        self.label_map = {
            "Normal": "Normal",
            "Abuse": "Abuse",
            "Arrest": "Arrest",
            "Arson": "Arson",
            "Assault": "Assault",
            "Burglary": "Burglary",
            "Explosion": "Explosion",
            "Fighting": "Fighting",
            "RoadAccidents": "RoadAccidents",
            "Robbery": "Robbery",
            "Shooting": "Shooting",
            "Shoplifting": "Shoplifting",
            "Stealing": "Stealing",
            "Vandalism": "Vandalism",
        }

    def extract_clip_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        features = []
        frames_batch = []
        frame_count = 0
        batch_size = 32

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames_batch.append(self.preprocess(pil_image))

                if len(frames_batch) == batch_size:
                    image_input = torch.stack(frames_batch).to(self.device)
                    with torch.no_grad():
                        batch_features = self.clip_model.encode_image(image_input)
                        batch_features /= batch_features.norm(dim=-1, keepdim=True)
                        features.append(batch_features.cpu().numpy())
                    frames_batch = []

            frame_count += 1

        if frames_batch:
            image_input = torch.stack(frames_batch).to(self.device)
            with torch.no_grad():
                batch_features = self.clip_model.encode_image(image_input)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
                features.append(batch_features.cpu().numpy())

        cap.release()

        if not features:
            raise ValueError("No frames extracted from video.")

        return np.concatenate(features, axis=0)

    def process_feat(self, feat, length):
        new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
        r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
        for i in range(length):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
        return new_feat

    def get_prompt_text(self):
        # index 0 为 Normal，后续为异常类别
        return list(self.label_map.values())

    @prompts(
        name="Detect Violence in Video",
        description="useful when you want to detect physical violence, fighting, assault, abuse, or aggressive behaviors in a video. "
        "The tool analyzes the video to identify violent events (e.g., Fighting, Assault, Shooting, Abuse). "
        "The input to this tool should be a comma-separated string containing two file paths: the first is the original video path, and the second is the pose-estimated video path. Do not include parameter names like 'video_path=' in the input. For example, a valid input is './Evo_data/videos/video_347487.mp4, videos/pose_video_347487.mp4'.",
    )
    def inference(self, inputs):
        paths = [p.strip() for p in inputs.split(",")]
        if len(paths) >= 2:
            original_video_path = paths[0]
            pose_video_path = paths[1]
        else:
            # 容错处理：如果Agent只传了一个路径
            original_video_path = paths[0]
            pose_video_path = None
            print(f"[Warning] Only one path provided: {original_video_path}")
        video_path = original_video_path
        if not os.path.exists(video_path):
            return f"Error: Video file not found at {video_path}"

        try:
            try:
                raw_features = self.extract_clip_features(video_path)
            except Exception as e:
                return f"Error extracting features from {video_path}: {str(e)}"

            # 2. 预处理
            processed_features = self.process_feat(raw_features, self.visual_length)
            visual = (
                torch.tensor(processed_features).unsqueeze(0).to(self.device).float()
            )  # (1, 256, 512)

            # 3. 准备输入
            lengths = torch.tensor([self.visual_length], dtype=torch.int)
            padding_mask = torch.zeros(1, self.visual_length).to(self.device)
            prompt_text = self.get_prompt_text()

            # 4. 模型推理
            with torch.no_grad():
                _, logits1, logits2 = self.model(
                    visual, padding_mask, prompt_text, lengths
                )

            # 5. 结果解析
            logits2 = logits2.squeeze(0)  # (T, Class_Num)
            probs = torch.softmax(logits2, dim=-1).cpu().numpy()

            normal_probs = probs[:, 0]
            anomaly_scores = 1 - normal_probs

            # 类别解析
            class_probs = probs[:, 1:]
            class_indices = np.argmax(class_probs, axis=1)
            class_names = list(self.label_map.values())[1:]

            max_score = np.max(anomaly_scores)
            threshold = 0.5  # 判定阈值

            if max_score > threshold:
                peak_idx = np.argmax(anomaly_scores)
                peak_class_idx = class_indices[peak_idx]
                peak_class_name = class_names[peak_class_idx]
                confidence = class_probs[peak_idx, peak_class_idx]

                if pose_video_path:
                    final_msg = f"Based on the analysis of the pose estimation video {pose_video_path}, violence detection identified {peak_class_name}."
                else:
                    final_msg = f"Violence detection identified {peak_class_name}."
            else:
                if pose_video_path:
                    final_msg = f"Based on the analysis of the pose estimation video {pose_video_path}, violence detection identified Normal."
                else:
                    final_msg = "Violence detection identified Normal."

            return final_msg

        except Exception as e:
            return f"Unknown Error in VadCLIP: {e}"
