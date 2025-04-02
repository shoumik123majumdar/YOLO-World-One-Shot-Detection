import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection
)

class EmbeddingExtractor:
    def __init__(self, clip_model_path='openai/clip-vit-base-patch32'):
        """
        Initialize embedding extractor with CLIP model
        
        Args:
            clip_model_path: Path to the CLIP model
        """
        # CLIP model initialization
        self.clip_model_path = clip_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_path)
        self.processor = AutoProcessor.from_pretrained(clip_model_path)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
        
        # Move models to GPU if available
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.vision_model.to(self.device)
        self.text_model.to(self.device)
        
        
    def extract_embedding_from_roi(self, image_path, roi, crop_save_path=""):
        """
        Extract CLIP embeddings from a region of interest.
    
        Args:
            image_path: Path to the image
            roi: Tuple of (x, y, width, height) for the bounding box
            crop_save_path: Optional path to save the cropped ROI
        
        Returns:
            Feature embedding tensor
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        # Extract the ROI with boundary checks
        x, y, w, h = roi
        x, y = max(0, x), max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
    
        if w <= 20 or h <= 20:
            print(f"Warning: ROI too small: {w}x{h}, skipping")
            return None
        
        roi_img = img[y:y+h, x:x+w]
    
        # Save cropped image if requested
        if crop_save_path:
            cv2.imwrite(crop_save_path, roi_img)
    
        # ** FIX: Ensure we have a 3-channel RGB image **
        if len(roi_img.shape) < 3 or roi_img.shape[2] != 3:
            print(f"Converting image with shape {roi_img.shape} to 3 channels")
        if len(roi_img.shape) == 2:  # Grayscale
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
        else:
            # Create a 3-channel image by repeating the data
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    
        # Convert BGR to RGB for PIL
        roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    
        # ** FIX: Resize to CLIP expected size to avoid issues **
        roi_img_rgb = cv2.resize(roi_img_rgb, (224, 224))
    
        # Convert to PIL for CLIP
        pil_image = Image.fromarray(roi_img_rgb).convert('RGB')  # Explicit RGB conversion
    
        # Extract embedding using CLIP vision model
        embedding = self.extract_clip_features(pil_image)
    
        return embedding

    def extract_clip_features(self, image):
        """
        Extract features using CLIP vision model.
    
        Args:
            image: PIL Image
        
        Returns:
            Normalized feature embedding
        """
        try:
            # ** FIX: Convert PIL to numpy and manually preprocess **
            img_array = np.array(image)
        
            # Convert to float and normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
        
            # Manual normalization using CLIP parameters
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        
            for i in range(3):
                img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
            
            # HWC to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
        
            # Convert to tensor with batch dimension
            image_tensor = torch.from_numpy(img_array).unsqueeze(0).to(self.device)
        
            # Get CLIP embeddings
            with torch.no_grad():
                # Send directly to the vision model, bypassing the processor
                outputs = self.vision_model(pixel_values=image_tensor)
            
            # Normalize the embeddings
            img_feats = outputs.image_embeds
            img_feats = F.normalize(img_feats, p=2, dim=-1)
        
            return img_feats
        
        except Exception as e:
            print(f"Error extracting CLIP features: {e}")
            import traceback
            traceback.print_exc()
            #    Return zero tensor as fallback
            return torch.zeros(1, self.vision_model.config.projection_dim, device=self.device)

    
    def select_roi(self, image_path):
        """
        Open a window for user to select a region of interest.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple (x, y, width, height) of the selected ROI
        """
        img = cv2.imread(image_path)
        roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows() 
        return roi
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1, embedding2: Embedding tensors
            
        Returns:
            Cosine similarity score
        """
        return F.cosine_similarity(embedding1, embedding2, dim=1).item()



if __name__ == "__main__":
    extractor = EmbeddingExtractor(clip_model_path='openai/clip-vit-base-patch32')
    extractor.view_model_layers()

    reference_image = "One_Shot_HVAC.jpg"
    reference_roi = extractor.select_roi(reference_image)
    reference_embedding = extractor.extract_embedding_from_roi(reference_image, reference_roi)
        
    test_image = "test_HVAC_batch/has_hvac_3.jpg"
    test_roi = extractor.select_roi(test_image)
    test_embedding = extractor.extract_embedding_from_roi(test_image, test_roi)
    
    similarity = extractor.calculate_similarity(reference_embedding, test_embedding)
    print(f"CLIP similarity: {similarity}")