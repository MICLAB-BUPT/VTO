import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from glob import glob
from tqdm import *
import pandas as pd

class VSClassification:
    def __init__(self, device):
        self.device = device
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = self.__load_labels()
        self.features_blobs = []
        self.model = self.__load_model().to(self.device)
        params = list(self.model.parameters())
        self.weight_softmax = params[-2].data
        self.weight_softmax[self.weight_softmax<0] = 0
        self.ratio = 0.1
        self.trasform = pd.read_csv('./transform.txt', header=None, index_col=0).to_dict()[1]

    def inference(self, input_path):
        video_cap = cv2.VideoCapture(input_path)
        nframes = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, img = video_cap.read()
        scene_idx_count = [0 for i in range(365)]
        if nframes > 50000:
            return 'other'
        for i in tqdm(range(nframes)):
            if not ret:
                break
            img = Image.fromarray(img)
            input_img = self.__returnTF()(img).unsqueeze(0).to(self.device)

            # forward pass
            with torch.no_grad():
                logit = self.model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                idx = torch.max(h_x, 0)[1]
                scene_idx_count[idx] += 1

            # output the IO prediction
            # io_image = np.mean(self.labels_IO[idx[:10]]) # vote for the indoor or outdoor
            # if io_image < 0.5:
            #     print('--TYPE OF ENVIRONMENT: indoor')
            # else:
            #     print('--TYPE OF ENVIRONMENT: outdoor')

            # output the prediction of scene category
            # print('--SCENE CATEGORIES:')
            # for i in range(0, 5):
            #     print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))

            # output the scene attributes
            # responses_attribute = self.W_attribute.dot(self.features_blobs[1])
            # idx_a = np.argsort(responses_attribute)
            # print('--SCENE ATTRIBUTES:')
            # print(', '.join([self.labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))

            ret, img = video_cap.read()
        scene = self.classes[scene_idx_count.index(max(scene_idx_count))]
        if scene in self.trasform.keys():
            scene = self.trasform[scene]
        else:
            scene = 'other'
        return scene

    def __recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.__recursion_change_bn(module1)
        return module
    
    def __load_labels(self):
        # prepare all the labels
        # scene category relevant
        file_name_category = 'categories_places365.txt'
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = 'IO_places365.txt'
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = 'labels_sunattribute.txt'
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'W_sceneattribute_wideresnet18.npy'
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute

    # def __hook_feature(self, module, input, output):
    #     self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))
    
    def __returnTF(self):
        # load the image transformer
        tf = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),   
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf
    
    def __load_model(self):
        # this model has a last conv feature map as 14x14

        model_file = 'wideresnet18_places365.pth.tar'

        import wideresnet
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        
        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
        for i, (name, module) in enumerate(model._modules.items()):
            module = self.__recursion_change_bn(model)
        model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        model.eval()
        # hook the feature extractor
        # features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
        # for name in features_names:
        #     model._modules.get(name).register_forward_hook(self.__hook_feature)
        return model
    

# if __name__ == '__main__':
#     device = torch.device('cuda')
#     model = VSClassification(device)
#     categories = []
#     count = []
#     video_label = {}
#     import pandas as pd
#     for index in range(10,40):
#         path = os.path.join('/home/mnt/zy/UCF-Crime/shuffle', str(index+1), '*.mp4')
#         f = glob(path)
#         print('第', index+1, '文件夹')
#         for i, video in enumerate(f):
#             print('vedio: (' + str(i+1) + '/' + str(len(f)) + ')')
#             scene = model.inference(video)
#             print(scene)
#             video = video.split('/')[-1]
#             video_label[video] = scene
#             if scene not in categories:
#                 categories.append(scene)
#                 count.append(1)
#             else:
#                 count[categories.index(scene)] += 1
#         df = pd.DataFrame(list(video_label.items()))
#         df.to_csv('./result/result{}.csv'.format(str(index+1)), header=None, index=None)
#         # empty
#         video_label = {}
#         print(categories)

if __name__ == '__main__':
    device = torch.device('cuda')
    model = VSClassification(device)
    categories = []
    count = []
    video_label = {}
    import pandas as pd

    path = os.path.join('/home/mnt/zy/UCF-Crime/test_data/*.mp4')
    f = glob(path)
    for i, video in enumerate(f):
        print('vedio: (' + str(i+1) + '/' + str(len(f)) + ')')
        scene = model.inference(video)
        print(scene)
        video = video.split('/')[-1]
        video_label[video] = scene
        if scene not in categories:
            categories.append(scene)
            count.append(1)
        else:
            count[categories.index(scene)] += 1
    df = pd.DataFrame(list(video_label.items()))
    df.to_csv('./result.csv', header=None, index=None)
    # empty
    video_label = {}
    print(categories)