import cv2
import numpy as np
import matplotlib.pyplot as plt
from CLIP_Embedding_Extractor import EmbeddingExtractor  # Import your updated CLIP-based EmbeddingExtractor

class RegionProposalGenerator:
    def __init__(self, embedding_extractor=None, similarity_threshold=0.83):
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
        
    def set_reference(self, image_path, roi=None, save_path = False):
        """
        Set the reference embedding from an image ROI.
        
        Args:
            image_path: Path to the reference image
            roi: Region of interest as (x, y, w, h), if None, user will select it
        """
        if roi is None:
            roi = self.embedding_extractor.select_roi(image_path)
        if save_path:
            

        # Use the new extract_embedding_from_roi method from your CLIP-based extractor
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
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return []
            
        # Setup selective search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        # Filter out tiny regions that might cause processing errors
        filtered_rects = []
        for rect in rects:
            x, y, w, h = rect
            if w >= 20 and h >= 20:  # Minimum size requirement
                filtered_rects.append(rect)
        
        return filtered_rects
    
    def detect_objects(self, image_path, max_proposals=50):
        """
        Detect objects in an image by comparing region proposals with reference embedding.
        
        Args:
            image_path: Path to the image
            max_proposals: Maximum number of proposals to evaluate
            
        Returns:
            List of detected regions as (x, y, w, h, similarity_score)
        """
        if self.reference_embedding is None:
            print("Error: Reference embedding not set. Call set_reference() first.")
            return []
        
        # Generate region proposals
        proposals = self.generate_proposals(image_path, max_proposals)
        
        # Extract embeddings and compare with reference
        detections = []
        
        print(f"Evaluating {len(proposals)} proposals...")
        for i, roi in enumerate(proposals):
            # Use the CLIP-based embedding extraction method
            embedding = self.embedding_extractor.extract_embedding_from_roi(image_path, tuple(roi))
            
            if embedding is not None:
                # Calculate similarity with reference embedding
                similarity = self.embedding_extractor.calculate_similarity(self.reference_embedding, embedding)
                
                if similarity > self.similarity_threshold:
                    detections.append((*roi, similarity))
        
        # Sort by similarity score (highest first)
        detections.sort(key=lambda x: x[4], reverse=True)
        
        print(f"Found {len(detections)} detections above threshold {self.similarity_threshold}")
        return detections
    
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
    
    def batch_process(self, image_dir, output_dir=None, iou_threshold=0.5, max_proposals=50):
        """
        Process a batch of images and detect objects in each.
        
        Args:
            image_dir: Directory containing images to process
            output_dir: Directory to save visualization results (optional)
            iou_threshold: IoU threshold for NMS
            max_proposals: Maximum number of proposals to evaluate per image
            
        Returns:
            Dictionary mapping image filenames to detection results
        """
        import os
        
        if self.reference_embedding is None:
            print("Error: Reference embedding not set. Call set_reference() first.")
            return {}
            
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        results = {}
        
        # Get all image files in the directory
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                      
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            print(f"\nProcessing {filename}...")
            
            # Detect objects
            detections = self.detect_objects(image_path, max_proposals)
            
            # Apply NMS
            filtered_detections = self.non_max_suppression(detections, iou_threshold)
            
            # Visualize and save results
            if output_dir:
                output_path = os.path.join(output_dir, f"detected_{filename}")
                self.visualize_detections(image_path, filtered_detections, 
                                          output_path=output_path, show=False)
            
            # Store results
            results[filename] = filtered_detections
            
        return results

if __name__ == "__main__":
    extractor = EmbeddingExtractor(clip_model_path='openai/clip-vit-base-patch32')
    region_generator = RegionProposalGenerator(extractor)
    
    reference_image = "One_Shot_HVAC.jpg"
    ref_roi = region_generator.set_reference(reference_image)
    
    test_image = "test_HVAC_batch/has_hvac_3.jpg"
    detections = region_generator.detect_objects(test_image, max_proposals=150)
    filtered_detections = region_generator.non_max_suppression(detections, iou_threshold=0.5)
    
    region_generator.visualize_detections(test_image, filtered_detections, output_path="detected_hvac.jpg")
    
    results = region_generator.batch_process("test_HVAC_batch", "output_detections")