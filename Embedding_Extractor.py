import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO

class EmbeddingExtractor:
    def __init__(self):
        self.model_path = "yolov8l-world.pt"
        self.reference_embedding = None
        self.layer_name = "model.22"
        self.model = YOLO(self.model_path)

        
    def extract_embedding_from_roi(self, image_path,roi, layer_name="model.22"):
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
    
        # Convert ROI to RGB (YOLOWorld expects RGB)
        roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    
        # Extract embedding from the ROI
        embedding = self.extract_features(roi_img_rgb, layer_name)
    
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
    
    def select_roi(self,image_path):
        img = cv2.imread(image_path)
        roi = cv2.selectROI("Select HVAC ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows() 
        return roi

if __name__ == "__main__":
    reference_image = "One_Shot_HVAC.jpg"
    extractor = EmbeddingExtractor()
    roi = extractor.select_roi(reference_image)
    reference_embedding = extractor.extract_embedding_from_roi(reference_image,roi)
    #Manually test if reference embedding cosine similarity is the same 
    test_image = "test_HVAC_batch/has_hvac_1.jpg"
    roi = extractor.select_roi(test_image)
    test_embedding = extractor.extract_embedding_from_roi(test_image,roi)

    similarity = F.cosine_similarity(reference_embedding, test_embedding, dim=1).item()
    print(similarity)
    