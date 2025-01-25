import faiss
import numpy as np
import json
import torchvision
from torchvision.transforms import v2 as transforms
import torch
import torch.nn as nn
from PIL import Image
import os
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.feature_extraction import create_feature_extractor





def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


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

index = faiss.read_index('index.faiss')

with open('master_list.json', 'r') as f:
    master_list = json.load(f)

image = Image.open('similarity_test/similarity_test_3.jpg')

model = get_model()

processed_image = get_transform(image)

with torch.no_grad():
    compare_embedding = model(processed_image)

compare_embedding = process_embeddings(compare_embedding)

print(compare_embedding.shape)
D, I = index.search(compare_embedding.detach().numpy(), 5)

print(I)
print(D)

top_indexes = I[0][0]
for i in I[0]:
    print(master_list[i])
