import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO

class EmbeddingExtractor:
    def __init__(self):
        self.model_path = "yolov8l-world.pt"
        self.reference_embedding = None
        self.layer_name = "model.23.cv4.1"
        #model.22 (layer)
        self.model = YOLO(self.model_path)

        
    def extract_embedding_from_roi(self, image_path,roi,crop_save_path=""):
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
        
        if crop_save_path:
            cv2.imwrite(crop_save_path,roi_img_rgb)
        

        # Extract embedding from the ROI
        embedding = self.extract_features(roi_img_rgb)
    
        return embedding

    
    
    def extract_features(self, image):
        """
        Extract features from a specific layer of the YOLO-World model.
    
        Args:
            image: Image as RGB numpy array
            layer_name: Name of the layer to extract features from
        
        Returns:
            Feature embedding tensor
        """
        
        layer_name = self.layer_name
            
        # Variable to store our embeddings
        embeddings_output = None
    
        # Hook function to capture the output of a specific layer
        def hook_fn(module, input, output):
            nonlocal embeddings_output
            embeddings_output = output

        #Take picture of cat and dog and shouldn't be the same
    
        # Find the specified layer and register our hook
        for name, module in self.model.model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break
    
    
        # Run inference to trigger the hook (without computing gradients)
        with torch.no_grad():
            self.model.predict(image, verbose=False)
    
        # Remove the hook
        hook.remove()

        print(f"Raw embedding shape: {embeddings_output.shape}")
        print(f"Sample values: {embeddings_output[0, :5]}")
    
        return embeddings_output
    
    def select_roi(self,image_path):
        img = cv2.imread(image_path)
        roi = cv2.selectROI("Select HVAC ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows() 
        return roi
    
    def view_model_layers(self):
        """
        Display all layers in the YOLO model with their types.
    
        This function prints each layer's name, type, and output shape if available.
        Useful for selecting which layer to extract embeddings from.
    
        Returns:
            dict: Dictionary mapping layer names to their module types
        """
        print("YOLO Model Layers:")
        print("-" * 80)
        print(f"{'Layer Name':<40} {'Type':<30} {'Parameters':<10}")
        print("-" * 80)
    
        layer_dict = {}
        for name, module in self.model.model.named_modules():
            if name:  # Skip empty names
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                layer_type = module.__class__.__name__
                layer_dict[name] = layer_type
                print(f"{name:<40} {layer_type:<30} {params:<10,}")
    
        print("\nRecommended feature extraction layers:")
        recommended = [
            layer for layer, type_ in layer_dict.items() 
            if "model." in layer and any(t in type_ for t in ["Conv", "C2f", "SPPF"])
        ]
    
        for i, layer in enumerate(recommended[-5:]):
            print(f" - {layer} ({layer_dict[layer]})")
    
        return layer_dict

if __name__ == "__main__":
    reference_image = "One_Shot_HVAC.jpg"
    extractor = EmbeddingExtractor()
    #extractor.view_model_layers()
    roi = extractor.select_roi(reference_image)

    reference_embedding = extractor.extract_embedding_from_roi(reference_image,roi,crop_save_path="reference_crop.jpg")
    #Manually test if reference embedding cosine similarity is the same 
    test_image = "test_HVAC_batch/has_hvac_2.jpg"
    roi = extractor.select_roi(test_image)
    test_embedding = extractor.extract_embedding_from_roi(test_image,roi,crop_save_path="test_crop.jpg")

    similarity = F.cosine_similarity(reference_embedding, test_embedding, dim=1).item()
    print(similarity)
    