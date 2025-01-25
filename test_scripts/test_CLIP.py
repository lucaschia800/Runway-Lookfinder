import json
from transformers import CLIPProcessor, CLIPModel 
import torch
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

if __name__ == "__main__":
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    # model.load_state_dict(torch.load('clip_model_v4.pth', map_location=torch.device('cpu'), weights_only= True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    image_batch = []
    for image in os.listdir('similarity_test'):
        print(image)
        image_batch.append(Image.open('similarity_test/' + image))

    text_batch = ['Clothes', 'Clothes', 'Clothes', 'Clothes', 'Clothes', 'Clothes', 'Clothes', 'Clothes']


    inputs = processor(images=image_batch, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    inputs_loss = processor(images=image_batch, text=text_batch, return_tensors="pt", padding=True, truncation = True)

    visual_parameters = [p for p in model.vision_model.parameters() if p.requires_grad]
    transformer_parameters = [p for p in model.text_model.parameters() if p.requires_grad]

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)


    print(embeddings.shape)

    
    similarity_cos = cosine_similarity(embeddings)
    similarity_euc = euclidean_distances(embeddings)
    print(similarity_cos)
    print(similarity_euc)