from __future__ import division

import os
import warnings
from collections import OrderedDict
from config import return_args, args  # Assuming config.py is present and correct
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms
from utils import setup_seed
# nni is not needed for single inference, but we keep it to avoid breaking imports
import nni
from nni.utils import merge_parameter
import util.misc as utils
import torch
import numpy as np
import cv2
import torch.nn as nn
from Networks.CDETR import build_model

# --- Global Transformations (from original script) ---
img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)


def show_map(out_pointes, frame, width, height, crop_size, num_h, num_w):
    """
    This function reconstructs the full-size point map and density map from the
    model's output on image patches. It also visualizes the results.
    (This function is unchanged from the original script)
    """
    kpoint_list = []
    confidence_list = []

    for i in range(len(out_pointes)):
        out_value = out_pointes[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_pointes[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
        k = np.zeros((crop_size, crop_size))

        # Get coordinate for each patch
        for j in range(len(out_point)):
            if out_value[j] < 0.25:  # Confidence threshold
                break
            # Ensure coordinates are within the 256x256 patch
            y = min(int(out_point[j][0]), crop_size - 1)
            x = min(int(out_point[j][1]), crop_size - 1)
            k[x, y] = 1

        kpoint_list.append(k)

    # Reconstruct the full point map from patches
    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous().view(num_h, crop_size, width).view(height, width).cpu().numpy()

    # Create density map
    density_map = gaussian_filter(kpoint.copy(), 6)
    density_map = density_map / (np.max(density_map) + 1e-8) * 255 # Add epsilon to avoid division by zero
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

    # Get coordinates and count
    pred_coor = np.nonzero(kpoint)
    count = len(pred_coor[0])

    # Create visualization maps
    point_map = np.zeros((int(kpoint.shape[0]), int(kpoint.shape[1]), 3), dtype="uint8") + 255  # White background
    for i in range(count):
        w = int(pred_coor[1][i])
        h = int(pred_coor[0][i])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)      # Draw black points on white map
        cv2.circle(frame, (w, h), 3, (0, 255, 50), -1)     # Draw green points on original image

    return point_map, density_map, frame, count

def process_single_image(model, image_path, args):
    """
    Loads a single image, processes it through the model, and saves the visualized output.
    """
    print(f"--- Processing image: {image_path} ---")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from path: {image_path}")
        return

    # Define target size (must be divisible by 256 for the patching logic)
    width = 1024
    height = 768
    
    frame = cv2.resize(frame, (width, height))
    ori_frame = frame.copy()

    # --- Pre-processing ---
    # Transform and normalize
    image = tensor_transform(frame)
    image = img_transform(image)

    # Tiling/Patching logic (from original script)
    # This splits the image into a batch of 256x256 patches
    img_w, img_h = image.shape[2], image.shape[1]
    num_w = int(img_w / 256)
    num_h = int(img_h / 256)

    image = image.view(3, num_h, 256, img_w).view(3, num_h, 256, num_w, 256)
    image = image.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, 256, 256).permute(1, 0, 2, 3)

    # --- Inference ---
    with torch.no_grad():
        image = image.cuda()
        outputs = model(image)

        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']

        prob = out_logits.sigmoid()
        # Get top predictions for each patch
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), args['num_queries'], dim=1)

        topk_points = topk_indexes // out_logits.shape[2]
        out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
        
        # Scale points from [0,1] to [0, 256]
        out_point = out_point * 256

        value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)
        crop_size = 256
        
        # --- Post-processing and Visualization ---
        kpoint_map, density_map, annotated_frame, count = show_map(value_points, frame, img_w, img_h, crop_size, num_h, num_w)

        # Combine the 4 images into a single output image
        res1 = np.hstack((ori_frame, kpoint_map))
        res2 = np.hstack((density_map, annotated_frame))
        res = np.vstack((res1, res2))

        # Add count text to the final composite image
        cv2.putText(res, "Count:" + str(count), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
        # Save the output image
        output_path = "output_crowd_counting.png"
        cv2.imwrite(output_path, res)
        
        print(f"Final Count: {count}")
        print(f"Output visualization saved to: {output_path}")


def main(args):
    # This part is mostly for model setup
    utils.init_distributed_mode(return_args)
    
    model, criterion, postprocessors = build_model(return_args)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    model.eval() # Set model to evaluation mode

    # --- Load pre-trained model checkpoint ---
    if args['pre'] and os.path.isfile(args['pre']):
        print("=> loading checkpoint '{}'".format(args['pre']))
        checkpoint = torch.load(args['pre'])
        
        # This logic for renaming keys is specific to the model and must be kept
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('bbox', 'point') 
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args['pre'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['pre']))
        print("=> Please provide a valid path to a pre-trained model.")
        return

    # --- Process the single image ---
    process_single_image(model, args['image_path'], args)


if __name__ == '__main__':
    # Instead of using nni, we define the parameters directly for inference.
    # You MUST update the 'pre' and 'image_path' values.
    params = {
        'pre': '/home/wyt/VIoTGPT/tools/CLTR_crowdcounting/ckpt/video_model.pth',  # !!! IMPORTANT: CHANGE THIS to your model path
        'image_path': '/home/wyt/VIoTGPT/0558.jpg', # The image to process
        'num_queries': 700, # This should match the model's configuration
        'device': 'cuda',
        # Add other necessary args from your config if the model requires them
    }

    # Merge with default args from config.py
    # This ensures all required arguments are present
    final_args = vars(merge_parameter(return_args, params))
    
    main(final_args)