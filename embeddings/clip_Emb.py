"""
CLIP-based Image Embedding Processor with Faiss Indexing

This script processes fashion designer data to generate CLIP image embeddings 
for garment crops and builds a Faiss index for efficient similarity search.
"""

import os
import json
import logging
from io import BytesIO
from typing import List, Tuple, Dict

import requests
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_clip_model(model_id: str, weights_path: str) -> Tuple[CLIPProcessor, CLIPModel]:
    """
    Load CLIP model with custom trained weights.
    
    Args:
        model_id: Pretrained model identifier
        weights_path: Path to fine-tuned model weights
    
    Returns:
        Tuple of CLIP processor and model
    """
    try:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        return processor, model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


def process_image_url(url: str) -> Image.Image:
    """
    Fetch and validate image from URL.
    
    Args:
        url: Image URL to download
    
    Returns:
        PIL Image object
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logging.warning(f"Failed to process image URL {url}: {str(e)}")
        return None


def extract_garment_crops(image: Image.Image, garments: Dict) -> List[Image.Image]:
    """
    Extract garment crops from image using bounding boxes.
    
    Args:
        image: Source PIL Image
        garments: Dictionary containing 'boxes' and 'labels'
    
    Returns:
        List of cropped garment images
    """
    crops = []
    for bbox in garments.get('boxes', []):
        left, top, right, bottom = bbox
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def process_designer_data(
    designer_data: List[Dict],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    batch_size: int = 8
) -> Tuple[np.ndarray, List[str], faiss.Index]:
    """
    Process designer data to generate embeddings and build Faiss index.
    
    Args:
        designer_data: List of designer dictionaries
        processor: CLIP processor
        model: CLIP model
        device: Torch device for computation
        batch_size: Processing batch size
    
    Returns:
        Tuple containing:
        - embeddings array
        - image URLs list
        - Faiss index
    """
    image_urls = []
    embeddings = np.empty((0, 512))
    index = faiss.IndexFlatL2(512)

    for designer in designer_data:
        for show in designer['Shows']:
            if not show.get('Looks'):
                continue

            for i in range(0, len(show['Looks']), batch_size):
                batch_looks = show['Looks'][i:i+batch_size]
                garment_images = []
                current_urls = []

                for look in batch_looks:
                    image = process_image_url(look['Look_Url'])
                    if not image:
                        continue

                    garments = look.get('Garments', {})
                    crops = extract_garment_crops(image, garments)
                    
                    if crops:
                        garment_images.extend(crops)
                        current_urls.extend([look['Look_Url']] * len(crops))

                if garment_images:
                    try:
                        inputs = processor(
                            images=garment_images, 
                            return_tensors="pt"
                        ).to(device)
                        
                        with torch.no_grad():
                            batch_embeddings = model.get_image_features(**inputs)
                        
                        batch_embeddings = batch_embeddings.cpu().numpy().squeeze()
                        if batch_embeddings.ndim == 1:
                            batch_embeddings = batch_embeddings.reshape(1, -1)
                            
                        index.add(batch_embeddings)
                        embeddings = np.vstack((embeddings, batch_embeddings))
                        image_urls.extend(current_urls)
                        
                        logging.info(f"Processed {len(current_urls)} garments from {designer['Designer']} - {show['Show Name']}")

                    except Exception as e:
                        logging.error(f"Error processing batch: {str(e)}")

    return embeddings, image_urls, index


def save_artifacts(
    embeddings: np.ndarray,
    image_urls: List[str],
    index: faiss.Index,
    embeddings_path: str = 'embeddings.npy',
    urls_path: str = 'image_urls.json',
    index_path: str = 'index.faiss'
):
    """Save processing artifacts to disk."""
    np.save(embeddings_path, embeddings)
    with open(urls_path, 'w') as f:
        json.dump(image_urls, f)
    faiss.write_index(index, index_path)
    logging.info(f"Saved artifacts to {embeddings_path}, {urls_path}, and {index_path}")


if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_ID = "openai/clip-vit-base-patch16"
    WEIGHTS_PATH = "clip_model_v4.pth"
    
    try:
        # Load data and model
        with open('letter_A.json') as f:
            designer_data = json.load(f)
            
        processor, model = load_clip_model(MODEL_ID, WEIGHTS_PATH)
        model.to(DEVICE).eval()

        # Process data
        embeddings, image_urls, index = process_designer_data(
            designer_data=designer_data,
            processor=processor,
            model=model,
            device=DEVICE,
            batch_size=8
        )

        # Save results
        save_artifacts(embeddings, image_urls, index)

    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise