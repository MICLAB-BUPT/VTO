import os
import torch
from decord import VideoReader, cpu
import matplotlib.pyplot as plt
from holmesvau.holmesvau_utils import load_model, generate, show_smapled_video

mllm_path = '/home/wyt/VIoTGPT/tools/HolmesVAU/ckpt'
sampler_path = '/home/wyt/VIoTGPT/tools/HolmesVAU/holmesvau/ATS/anomaly_scorer.pth'
device = torch.device('cuda:0')
model, tokenizer, generation_config, sampler = load_model(mllm_path, sampler_path, device)

video_path = "/home/wyt/VIoTGPT/tools/HolmesVAU/examples/robbery.mp4"
prompt = "Could you specify the anomaly events present in the video?"
pred, history, frame_indices, anomaly_score = generate(video_path, prompt, model, tokenizer, generation_config, sampler, select_frames=12, use_ATS=True)
print('\nUser:', prompt, '\nHolmesVAU:', pred)


