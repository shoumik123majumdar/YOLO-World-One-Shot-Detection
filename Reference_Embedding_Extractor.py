import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os

class ReferenceEmbeddingExtractor:
    def __init__(self, image_path=""):
        self.image_path = image_path
        self.model_path = "yolov8l-world.pt"
        self.model = self.load_yolo_model(self.model_path)
        self.reference_embedding = None
        self.layer_name = "model.22"
        
    def load_yolo_model(self, model_path="yolov8l-world.pt"):
        """Load the YOLO-World model."""
        print(f"Loading YOLO-World model from {model_path}...")
        model = YOLO(model_path)
    
        model.set_classes([
            "outdoor air conditioning unit",
            "HVAC unit",
            "air conditioner",
            "heat pump"
        ])
    
        return model

    def extract_embedding_from_roi(self, image_path=None, roi=None, layer_name=None, visualize=True):
        """
        Extract features from a manually specified region of interest.
    
        Args:
            image_path: Path to the image (uses self.image_path if None)
            roi: Tuple of (x, y, width, height) for the bounding box
            layer_name: Name of the layer to extract features from
            visualize: Whether to display the ROI
        
        Returns:
            Feature embedding tensor
        """
        # Use provided parameters or defaults from class
        if image_path is None:
            image_path = self.image_path
        
        if layer_name is None:
            layer_name = self.layer_name
            
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None
    
        # Extract the ROI
        x, y, w, h = roi
        roi_img = img[y:y+h, x:x+w]
    
        # Visualize the ROI if requested
        if visualize:
            # Create a copy of the original image for drawing
            img_with_roi = img.copy()
        
            # Draw the bounding box on the copy
            cv2.rectangle(img_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
            # Convert BGR to RGB for matplotlib
            img_with_roi_rgb = cv2.cvtColor(img_with_roi, cv2.COLOR_BGR2RGB)
            roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        
            # Display the image with ROI
            ax1.imshow(img_with_roi_rgb)
            ax1.set_title('Image with ROI')
            ax1.axis('off')
        
            # Display the cropped ROI
            ax2.imshow(roi_img_rgb)
            ax2.set_title('Cropped ROI')
            ax2.axis('off')
        
            plt.tight_layout()
            plt.show()
    
        # Convert ROI to RGB (YOLOWorld expects RGB)
        roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    
        # Extract embedding from the ROI
        embedding = self.extract_features(roi_img_rgb, layer_name)
    
        if embedding is not None:
            print(f"Successfully extracted embedding with shape: {embedding.shape}")
            self.reference_embedding = embedding
    
        return embedding

    def extract_features(self, image, layer_name=None):
        """
        Extract features from a specific layer of the YOLO-World model.
    
        Args:
            image: Image as RGB numpy array
            layer_name: Name of the layer to extract features from
        
        Returns:
            Feature embedding tensor
        """
        if layer_name is None:
            layer_name = self.layer_name
            
        # Variable to store our embeddings
        embeddings_output = None
    
        # Hook function to capture the output of a specific layer
        def hook_fn(module, input, output):
            nonlocal embeddings_output
            embeddings_output = output
    
        # Find the specified layer and register our hook
        layer_found = False
        for name, module in self.model.model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                layer_found = True
                break
    
        if not layer_found:
            print(f"Layer '{layer_name}' not found in the model.")
            return None
    
        # Run inference to trigger the hook (without computing gradients)
        with torch.no_grad():
            self.model.predict(image, verbose=False)
    
        # Remove the hook
        hook.remove()
    
        # Check if we successfully captured embeddings
        if embeddings_output is None:
            print(f"Failed to extract embeddings from layer {layer_name}")
            return None
    
        # Process the extracted embeddings
        if isinstance(embeddings_output, torch.Tensor):
            if embeddings_output.dim() > 2:
                # This is important: we're taking feature maps (NCHW format) and averaging
                # across the spatial dimensions (H,W) to get a single feature vector
                embeddings = F.adaptive_avg_pool2d(embeddings_output, 1).squeeze(-1).squeeze(-1)
                # Normalize for better similarity comparison
                embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings
    
        return embeddings_output

    def print_model_layers(self):
        """Print all layer names in the model for reference."""
        print("\nYOLO-World Model Layers:")
        print("-" * 50)
        for name, module in self.model.model.named_modules():
            print(f"Layer: {name} - Type: {type(module).__name__}")
        print("-" * 50)

    def compare_embeddings(self, query_embedding=None, reference_embedding=None):
        """Compare two embeddings using cosine similarity."""
        if reference_embedding is None:
            reference_embedding = self.reference_embedding
            
        if reference_embedding is None:
            print("No reference embedding available. Extract one first.")
            return None
            
        similarity = F.cosine_similarity(reference_embedding, query_embedding, dim=1).item()
        print(f"Similarity score: {similarity:.4f}")
        return similarity

    def run_yolo_prediction(self, image_path=None):
        """Run YOLO-World detection to see what objects it finds."""
        if image_path is None:
            image_path = self.image_path
            
        results = self.model.predict(image_path, verbose=True)
        
        # Get the image with detections
        annotated_img = results[0].plot()
        
        # Convert to RGB for matplotlib
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_img_rgb)
        plt.title("YOLO-World Detections")
        plt.axis('off')
        plt.show()
        
        return results
        
    def extract_and_compare(self, test_image_path, test_roi, visualize=True):
        """Extract embedding from test image ROI and compare with reference."""
        test_embedding = self.extract_embedding_from_roi(
            image_path=test_image_path,
            roi=test_roi,
            visualize=visualize
        )
        
        if test_embedding is not None and self.reference_embedding is not None:
            similarity = self.compare_embeddings(test_embedding)
            return similarity, test_embedding
        return None, test_embedding


# Example usage
if __name__ == "__main__":
    # 1. Initialize the extractor with reference image
    reference_image = "test_HVAC_batch/has_hvac_1.jpg"
    extractor = ReferenceEmbeddingExtractor(reference_image)
    
    # Optional: Print model layers to see all available layers
    # extractor.print_model_layers()
    
    # Optional: See what YOLO-World detects in the reference image
    # extractor.run_yolo_prediction()
    
    # 2. Manually set the bounding box coordinates (x, y, width, height)
    # Example: roi = (100, 50, 200, 150)  # Adjust these values for your image
    roi = (100, 50, 200, 150)  # Replace with actual coordinates of your HVAC unit
    
    # 3. Extract embedding from the ROI
    reference_embedding = extractor.extract_embedding_from_roi(roi=roi)
    
    # 4. Now you can use this embedding for comparison with other regions
    # Example with a test image:
    # test_image = "test_HVAC_batch/has_hvac_2.jpg"
    # test_roi = (120, 60, 180, 140)  # Another region to compare
    # similarity, _ = extractor.extract_and_compare(test_image, test_roi)
    # print(f"Final similarity score: {similarity}")