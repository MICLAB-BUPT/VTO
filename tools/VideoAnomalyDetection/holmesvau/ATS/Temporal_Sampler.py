import numpy as np
from scipy import interpolate
import torch
from .anomaly_scorer import URDMU

class Temporal_Sampler():
    def __init__(self, ckpt_path, device):
        self.device = device
        self.anomaly_scorer = URDMU().to(device)
        self.anomaly_scorer.load_state_dict(torch.load(ckpt_path,map_location=device))
        self.tau = 0.1

    def get_anomaly_scores(self, pixel_values, model):
        def batchify(pixel_values, batch_size=16):
            N = pixel_values.shape[0]
            batches = []  
            for i in range(0, N, batch_size):   
                batch_end = min(i + batch_size, N)  
                batch = pixel_values[i:batch_end, :, :, :]  
                batches.append(batch)   
            return batches
        with torch.no_grad():
            pixel_values = pixel_values.to(torch.bfloat16)
            pixel_values_batched = batchify(pixel_values)
            cls_tokens = []
            for b, data in enumerate(pixel_values_batched):
                data = data.to(self.device)
                vit_embeds = model.vision_model(pixel_values=data, output_hidden_states=False, return_dict=True).last_hidden_state
                cls_token = vit_embeds[:, 0, :] #.cpu().detach()
                cls_tokens.append(cls_token)
                print("Extracted {}/{}".format(b, len(pixel_values_batched)), end='\r')
            vid_feats = torch.cat(cls_tokens, dim=0).to(torch.float32).unsqueeze(0)
            # print(pixel_values.shape, vid_feats.shape)
            anomaly_scores = self.anomaly_scorer(vid_feats)['anomaly_scores']
            anomaly_scores = anomaly_scores[0].detach().cpu().numpy()
        return anomaly_scores
    
    def density_aware_sample(self, pixel_values, model, select_frames=16):
        '''
        pixel_values: [T, C, H, W]
        '''
        anomaly_score = self.get_anomaly_scores(pixel_values, model)
        num_frames = anomaly_score.shape[0]
        if num_frames <= select_frames or sum(anomaly_score) < 1:
            sampled_idxs = list(np.rint(np.linspace(0, num_frames-1, select_frames)))
            return anomaly_score, sampled_idxs
        else:
            scores = [score + self.tau for score in anomaly_score]
            score_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(scores)), axis=0)
            max_score_cumsum = np.round(score_cumsum[-1]).astype(int)
            f_upsample = interpolate.interp1d(score_cumsum, np.arange(num_frames+1), kind='linear', axis=0, fill_value='extrapolate')
            scale_x = np.linspace(1, max_score_cumsum, select_frames)
            sampled_idxs = f_upsample(scale_x)
            sampled_idxs = [min(num_frames-1, max(0, int(idx))) for idx in sampled_idxs]
            return anomaly_score, sampled_idxs

    