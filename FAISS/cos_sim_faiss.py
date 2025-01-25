
from torchvision.transforms import v2 as transforms
from PIL import Image
import PIL
import os
import json
import requests
from io import BytesIO
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as Fu
import torch.nn as nn
import faiss
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.feature_extraction import create_feature_extractor

def process_embeddings(embeddings):
    embeddings = embeddings['output']
    embeddings = embeddings.squeeze().detach().cpu().numpy()

    return embeddings




def get_transform(train):
    if train:
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToDtype(torch.float32, scale=True),  # Convert image to tensor and set dtype
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=224),
            transforms.ToDtype(torch.float32, scale=True),  # Convert image to tensor and set dtype
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def get_model(num_classes = 46):
    # Load pre-trained model
    model = resnet101(weights=ResNet101_Weights.DEFAULT)

    # Replace the classifier with a new one for transfer learning
    model.fc = nn.Linear(in_features = 2048, out_features = num_classes, bias = True)




    model.load_state_dict(torch.load("resnet_101_finetune.pth", map_location=torch.device('cpu'), weights_only=True))

    return_nodes = {
    'avgpool' : 'output',
    }

    feature_extractor = create_feature_extractor(model, return_nodes)
    feature_extractor.eval()



    return feature_extractor



def embeddings_letter(letter, model, transforms, data, batch_size, master_embeddings):

    for designer in data:
            if designer['Designer'].startswith(letter):

                for show in designer['Shows']:

                    if show['Looks'] is None:

                        continue

                    for i in range(0, len(show['Looks']), batch_size):
                        garments_cropped = []

                        found = False

                        for look in show['Looks'][i: i + batch_size]:
                            boxes = []
                            labels = []

                            img_url = look['Look_Url']
                            try:
                                r = requests.get(img_url)
                            except requests.exceptions.RequestException as e:
                                print('Request Error')
                                continue
                            if r.status_code == 200:
                                try:

                                    img = Image.open(BytesIO(r.content))

                                except PIL.UnidentifiedImageError:
                                        print('Unidentified Image Error')

                                        continue
                            else:
                                    continue
                            if 'boxes' in look['Garments']:

                                found = True
                                for bbox in look['Garments']['boxes']:
                                    boxes.append(bbox)
                                for label in look['Garments']['labels']:
                                    labels.append(label)
                                for bbox in boxes:

                                    master_list.append(img_url)
                                    left, top, right, bottom = bbox
                                    img_cropped = img.crop((left, top, right, bottom))
                                    garments_cropped.append(img_cropped)

                        print(designer['Designer'], show['Show Name'])
                        print(i)
                        if found and len(garments_cropped) > 0:
                            garments_processed = transforms(garments_cropped)
                            if len(garments_processed) == 0:
                                continue

                            garments_final = torch.stack(garments_processed).to(device)
                            print('modeling')
                            with torch.no_grad():
                                embeddings = model(garments_final)


                            processed_embeddings = process_embeddings(embeddings)


                            if processed_embeddings.ndim == 1:

                                processed_embeddings = processed_embeddings.reshape(1, -1)
                            index.add(processed_embeddings)
                            master_embeddings = np.vstack((master_embeddings, processed_embeddings))

    return master_embeddings

if __name__ == "__main__":
    with open('data_Proj/letter_A.json', 'r') as f:
        data = json.load(f)

    model = get_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    batch_size  = 40


    master_list = []

    master_embeddings = np.empty((0, 2048))
    index = faiss.IndexFlatL2(2048)

    master_embeddings = embeddings_letter('A', model, get_transform(train=False), data, batch_size, master_embeddings)

    with open('master_list.json', 'w') as f:
        json.dump(master_list, f)

    faiss.write_index(index, 'index.faiss')
    np.save('master_embeddings.npy', master_embeddings)
