import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os
from sklearn.preprocessing import normalize



def preprocess_image(img_path):
    """Load and preprocess an image."""
    images_processed = []
    for image in img_path:
        img = Image.open(image).convert('RGB')
        img = preprocess(img)
        images_processed.append(img)
        images = torch.stack(images_processed)
    return images



def imshow(tensor, mean, std):

    # Clone the tensor to avoid modifying the original
    tensor = tensor.clone()

    # Remove the batch dimension
    tensor = tensor.squeeze(0)

    # Denormalize the tensor
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # Convert the tensor to a NumPy array
    img = tensor.numpy()

    # Transpose the dimensions to (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # Clip the image to the valid range [0, 1]
    img = np.clip(img, 0, 1)

    # Plot the image
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    plt.savefig('test_2.png') 


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
])



model = resnet50(weights = ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(in_features = 2048, out_features = 26, bias = True)
model.load_state_dict(torch.load('resnet_50_finetune.pth', map_location=torch.device('cpu')))


model.eval()
 

images = []
for image_path in os.listdir("similarity_test"):
    images.append(os.path.join("similarity_test" , image_path))

images_processed = preprocess_image(images)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# # Plot the processed image
# imshow(image_processed, mean=imagenet_mean, std=imagenet_std)



return_nodes = {
    'avgpool' : 'output',
    'layer4.2.conv3' : 'conv',
    'layer4.2.bn3' : 'batchnorm'
}

     
feature_extractor = create_feature_extractor(model, return_nodes)
feature_extractor.eval()

# activations = {}

# def get_activation(name):
#     def hook(model, input, output):
#         activations[name] = output.detach()
#     return hook

# hook = model.layer4[2].relu.register_forward_hook(get_activation('layer4_2_relu'))

with torch.no_grad():
    embeddings = feature_extractor(images_processed)



def process_embeddings(embeddings):
    embeddings = embeddings['output']
    embeddings = embeddings.squeeze().detach().numpy()

    # normalized_embeddings = normalize(embeddings, norm='l2')


    return embeddings


processed_embeddings = process_embeddings(embeddings)


similarity_cos = cosine_similarity(processed_embeddings)
similarity_euc = euclidean_distances(processed_embeddings)
print(similarity_cos)
print(similarity_euc)
print(images)

