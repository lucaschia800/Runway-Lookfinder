"""
Image Similarity Search System

This system combines CLIP and ResNet models with FAISS for efficient similarity search.
"""

import faiss
import json
import logging
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2 as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ImageProcessor:
    """Handles image preprocessing for different models."""
    
    @staticmethod
    def clip_transform():
        """CLIP-specific image transformations."""
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def resnet_transform():
        """ResNet-specific image transformations."""
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class FeatureExtractor:
    """Manages feature extraction models and operations."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_processor = None
        self.clip_model = None
        self.resnet_extractor = None
        self.faiss_index = None
        self.metadata = None

    def load_resnet(self, weights_path: str):
        """Load ResNet feature extractor with custom weights."""
        try:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(2048, 26)
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval().to(self.device)
            self.resnet_extractor = model
            logger.info("ResNet feature extractor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ResNet: {str(e)}")
            raise

    def load_clip(self, model_id: str, weights_path: str):
        """Load CLIP model with custom weights."""
        try:
            self.clip_model = CLIPModel.from_pretrained(model_id)
            self.clip_processor = CLIPProcessor.from_pretrained(model_id)
            self.clip_model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            self.clip_model.eval().to(self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            raise

    def load_faiss_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and associated metadata."""
        try:
            self.faiss_index = faiss.read_index(index_path)
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded FAISS index with {len(self.metadata)} entries")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise

    def get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for an image."""
        try:
            inputs = self.clip_processor(
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                
            return features.cpu().numpy().squeeze()
        except Exception as e:
            logger.error(f"Error generating CLIP embedding: {str(e)}")
            raise

    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """Search for similar items in the FAISS index."""
        if self.faiss_index is None or self.metadata is None:
            raise ValueError("FAISS index and metadata must be loaded first")
            
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)
        return [
            {
                "id": int(idx),
                "distance": float(dist),
                "metadata": self.metadata[int(idx)]
            }
            for idx, dist in zip(indices[0], distances[0])
        ]


def main():
    """Example usage of the similarity search system."""
    try:
        # Initialize feature extractor
        extractor = FeatureExtractor()
        
        # Load models and data
        extractor.load_clip(
            model_id="openai/clip-vit-base-patch16",
            weights_path="clip_model_v4.pth"
        )
        extractor.load_faiss_index(
            index_path="index.faiss",
            metadata_path="master_list.json"
        )

        # Process query image
        query_image = Image.open("similarity_test/OL_jacket_test.jpg")
        embedding = extractor.get_clip_embedding(query_image)
        
        # Perform similarity search
        results = extractor.search_similar(embedding, k=5)
        
        # Display results
        for result in results:
            print(f"ID: {result['id']} | Distance: {result['distance']:.4f}")
            print(f"Metadata: {result['metadata']}\n")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()