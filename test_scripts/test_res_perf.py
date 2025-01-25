import torch
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.transforms import v2 as transforms
import torch.nn as nn
from PIL import Image


def get_model(num_classes = 46):
    # Load pre-trained model
    model = resnet101(weights=ResNet101_Weights.DEFAULT)

    # Replace the classifier with a new one for transfer learning
    model.fc = nn.Linear(in_features = 2048, out_features = num_classes, bias = True)



    model.load_state_dict(torch.load("resnet_101_finetune.pth", map_location=torch.device('cpu'), weights_only=True))


    return model

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


image = Image.open("similarity_test/similarity_test_4.jpg")
image = preprocess(image)
model = get_model()
model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0))


print(output['output'])

