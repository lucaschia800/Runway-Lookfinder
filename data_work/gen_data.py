from PIL import Image
import PIL
import os
import json
import requests
from io import BytesIO
import pprint
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
import pprint
import torch.nn.functional as Fu


def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_detections_per_img=8,
                                       box_score_thresh = 0.83)

    # Replace the classifier with a new one for transfer learning
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def transform(batch):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),  # Convert image to tensor and set dtype
    ])
    transformed = []
    for img in batch:
        transformed.append(transform(img))

    return transformed

def predictions_letter(letter, master_dict, device, batch_size):

    for designer in master_dict:
        print(designer['Designer'])
        if designer['Designer'].startswith(letter):
            for show in designer['Shows']:
                if show['Looks'] is None:
                    continue
                for idx_outer, i in range(0, len(show['Looks']), batch_size):
                    batch = show['Looks'][i:i + batch_size]
                    image_batch = []
                    for look in batch:
                        look_curr = look['Look_Url']
                        try:
                            r = requests.get(look_curr)
                        except requests.exceptions.RequestException as e:
                            print('Request Error')
                            continue
                        if r.status_code == 200:
                            try:
                                img = Image.open(BytesIO(r.content))
                                image_batch.append(img)
                            except PIL.UnidentifiedImageError:
                                print('Unidentified Image Error')
                                continue
                        else:
                            continue
                    batch_transformed = transform(image_batch)
                    print(len(batch_transformed))
                    print(designer['Designer'], show['Show Name'])
                    batch_transformed = list(image.to(device) for image in batch_transformed)
                    with torch.no_grad():
                        predictions = model(batch_transformed)


                    for idx, prediction in enumerate(predictions):
                        boxes = prediction['boxes'].cpu().numpy()
                        labels = prediction['labels'].cpu().numpy()
                        scores = prediction['scores'].cpu().numpy()

                        show['Looks'][idx_outer * batch_size + idx]['Garments'] = {
                            'labels' : labels.tolist(),
                            'boxes' : boxes.tolist(), 
                            'scores' : scores.tolist()
                        }

    return master_dict



with open('all_photos.json', 'r') as file:
    master_dict = json.load(file)

rows = []
for designer in master_dict:
    designer_curr = designer['Designer']
    for show in designer['Shows']:
        show_curr = show['Show Name']
        if show['Looks'] is None:
            continue
        for idx, look_curr in enumerate(show['Looks']):
            look_dict = {
                'Look_Number' : idx + 1,
                'Look_Url' : look_curr,
                'Garments' : {}
                
            }
            
            show['Looks'][idx] = look_dict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model(10)
model.load_state_dict(torch.load('rcnn_finetune.pth'))
model.eval()
model.to(device)


with open('letter_A.json', 'w') as json_file:
    json.dump(dict, json_file)

