import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
from Embedding_Extractor import EmbeddingExtractor  # Import your existing module

class RegionProposalGenerator:
    def __init__(self, embedding_extractor=None, similarity_threshold=0.9):
        """
        Initialize the Region Proposal Generator.
        
        Args:
            embedding_extractor: An instance of EmbeddingExtractor
            similarity_threshold: Threshold for cosine similarity (default: 0.7)
        """
        # Initialize the embedding extractor
        self.embedding_extractor = embedding_extractor if embedding_extractor else EmbeddingExtractor()
        
        # Store the similarity threshold
        self.similarity_threshold = similarity_threshold
        self.reference_embedding = None
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        
    def set_reference(self, image_path, roi=None):
        """
        Set the reference embedding from an image ROI.
        
        Args:
            image_path: Path to the reference image
            roi: Region of interest as (x, y, w, h), if None, user will select it
        """
        if roi is None:
            roi = self.embedding_extractor.select_roi(image_path)
            
        self.reference_embedding = self.embedding_extractor.extract_embedding_from_roi(image_path, roi)
        print(f"Reference embedding set from ROI: {roi}")
        return roi
        
    def generate_proposals(self, image_path, max_proposals=50):
        """
        Generate region proposals for an image.
        
        Args:
            image_path: Path to the image
            max_proposals: Maximum number of proposals to return
            
        Returns:
            List of proposed regions as (x, y, w, h)
        """
        img = cv2.imread(image_path)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()[:max_proposals] # Get Top 50 Object Proposals
        return rects
    
    def detect_objects(self, image_path, max_proposals=50):
        """
        Detect objects in an image by comparing region proposals with reference embedding.
        
        Args:
            image_path: Path to the image
            method: Method to generate proposals ('selective_search' or 'grid')
            max_proposals: Maximum number of proposals to return
            
        Returns:
            List of detected regions as (x, y, w, h, similarity_score)
        """
        if self.reference_embedding is None:
            print("Error: Reference embedding not set. Call set_reference() first.")
            return []
        
        # Generate region proposals
        proposals = self.generate_proposals(image_path,max_proposals)
        
        # Extract embeddings and compare with reference
        detections = []
        
        print(f"Evaluating {len(proposals)} proposals...")
        for i, roi in enumerate(proposals):
            embedding = self.embedding_extractor.extract_embedding_from_roi(image_path, roi)
            
            if embedding is not None:
                similarity = F.cosine_similarity(self.reference_embedding, embedding, dim=1).item()
                
                if similarity > self.similarity_threshold:
                    detections.append((*roi, similarity))
        
        detections.sort(key=lambda x: x[4], reverse=True)
        
        print(f"Found {len(detections)} detections above threshold {self.similarity_threshold}")
        return detections #Return top 3 detections
    
    def non_max_suppression(self, detections, iou_threshold=0.5):
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            detections: List of detections as (x, y, w, h, score)
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
            
        # Convert to (x1, y1, x2, y2, score) format
        boxes = np.array([[d[0], d[1], d[0] + d[2], d[1] + d[3], d[4]] for d in detections])
        
        # Extract coordinates and scores
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        # Calculate areas
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with rest of the boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        # Convert back to original format
        filtered_detections = [detections[i] for i in keep]
        
        print(f"After NMS: {len(filtered_detections)} detections")
        return filtered_detections
    
    def visualize_detections(self, image_path, detections, output_path=None, show=True):
        """
        Visualize detected regions on the image.
        
        Args:
            image_path: Path to the image
            detections: List of detections as (x, y, w, h, similarity_score)
            output_path: Path to save the visualization image (optional)
            show: Whether to display the image
            
        Returns:
            Visualization image
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None
            
        # Make a copy for visualization
        vis_img = img.copy()
        
        # Draw each detection
        for i, (x, y, w, h, score) in enumerate(detections):
            # Draw rectangle
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
            
            # Add label with score
            label = f"{score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(vis_img, label, (x, y - 5), font, font_scale, color, thickness)
        
        # Convert to RGB for matplotlib
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Display using matplotlib
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_img_rgb)
            plt.axis('off')
            plt.title(f"Detected Objects ({len(detections)})")
            plt.show()
            
        # Save if output_path provided
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"Visualization saved to {output_path}")
            
        return vis_img_rgb

# Example usage
if __name__ == "__main__":
    # Initialize the modules
    extractor = EmbeddingExtractor()
    region_generator = RegionProposalGenerator(extractor, similarity_threshold=0.9)
    
    # Set reference from an image
    reference_image = "One_Shot_HVAC.jpg"
    ref_roi = region_generator.set_reference(reference_image)
    
    # Detect objects in a test image
    test_image = "test_HVAC_batch/has_hvac_2.jpg"
    detections = region_generator.detect_objects(test_image,max_proposals=50)
    
    # Apply NMS to remove overlapping detections
    filtered_detections = region_generator.non_max_suppression(detections, iou_threshold=0.5)
    
    # Visualize results
    region_generator.visualize_detections(test_image, filtered_detections, output_path="detected_hvac.jpg")