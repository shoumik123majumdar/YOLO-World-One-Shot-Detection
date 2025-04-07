import cv2
import numpy as np
import matplotlib.pyplot as plt
from CLIP_Embedding_Extractor import EmbeddingExtractor  # Import your updated CLIP-based EmbeddingExtractor
from sklearn.cluster import DBSCAN

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
        self.reference_roi = None
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        
    def set_reference(self, image_path, roi=None, save_path=None):
        """
        Set the reference embedding from an image ROI.
        
        Args:
            image_path: Path to the reference image
            roi: Region of interest as (x, y, w, h), if None, user will select it
            save_path: Path to save the reference crop (optional)
        """
        if roi is None:
            roi = self.embedding_extractor.select_roi(image_path)
        
        # Store reference ROI dimensions for proposal filtering
        self.reference_roi = roi

        # Use the new extract_embedding_from_roi method from your CLIP-based extractor
        self.reference_embedding = self.embedding_extractor.extract_embedding_from_roi(image_path, roi)
        print(f"Reference embedding set from ROI: {roi}")
        
        # Save reference crop if requested
        if save_path:
            img = cv2.imread(image_path)
            x, y, w, h = roi
            reference_crop = img[y:y+h, x:x+w]
            cv2.imwrite(save_path, reference_crop)
            
        return roi
        
    def filter_proposals(self, img, rects, max_proposals=100):
        """
        Filter region proposals to reduce the number of candidates before embedding comparison.
        
        Args:
            img: Input image as numpy array
            rects: List of region proposals as (x, y, w, h)
            max_proposals: Maximum number of proposals to return
            
        Returns:
            Filtered list of region proposals
        """
        if not rects:
            return []
        
        # Convert to numpy array for easier manipulation
        proposals = np.array(rects)
        scores = np.ones(len(proposals))  # Placeholder scores for NMS
        
        # 1. Filter by size - eliminate very small or very large regions
        img_area = img.shape[0] * img.shape[1]
        areas = proposals[:, 2] * proposals[:, 3]
        min_area = img_area * 0.001  # Minimum 0.1% of image area
        max_area = img_area * 0.5    # Maximum 50% of image area
        size_mask = (areas > min_area) & (areas < max_area)
        proposals = proposals[size_mask]
        scores = scores[size_mask]
        
        if len(proposals) == 0:
            return []
        
        # 2. Filter by aspect ratio if reference ROI is available
        if self.reference_roi is not None:
            ref_width, ref_height = self.reference_roi[2], self.reference_roi[3]
            ref_aspect = ref_width / max(ref_height, 1)  # Avoid division by zero
            
            # Calculate aspect ratios of proposals
            aspects = proposals[:, 2] / np.maximum(proposals[:, 3], 1)
            
            # Keep proposals with similar aspect ratio (within 50% tolerance)
            aspect_mask = (aspects > ref_aspect * 0.5) & (aspects < ref_aspect * 1.5)
            proposals = proposals[aspect_mask]
            scores = scores[aspect_mask]
            
            if len(proposals) == 0:
                return []
        
        # 3. Apply Non-Maximum Suppression to reduce overlapping proposals
        # Convert to format expected by NMS: [x1, y1, x2, y2]
        boxes_for_nms = np.zeros((len(proposals), 5))
        boxes_for_nms[:, 0] = proposals[:, 0]
        boxes_for_nms[:, 1] = proposals[:, 1]
        boxes_for_nms[:, 2] = proposals[:, 0] + proposals[:, 2]
        boxes_for_nms[:, 3] = proposals[:, 1] + proposals[:, 3]
        boxes_for_nms[:, 4] = scores
        
        # Apply custom NMS implementation
        keep_indices = self._custom_nms(boxes_for_nms, iou_threshold=0.3)
        filtered_proposals = proposals[keep_indices]
        
        # 4. Cluster similar proposals and keep representatives
        if len(filtered_proposals) > max_proposals * 2:
            # Use proposal centers and sizes as features for clustering
            features = np.zeros((len(filtered_proposals), 4))
            features[:, 0] = filtered_proposals[:, 0] + filtered_proposals[:, 2] / 2  # center x
            features[:, 1] = filtered_proposals[:, 1] + filtered_proposals[:, 3] / 2  # center y
            features[:, 2] = filtered_proposals[:, 2]  # width
            features[:, 3] = filtered_proposals[:, 3]  # height
            
            # Normalize features
            feature_ranges = features.max(axis=0) - features.min(axis=0)
            feature_ranges[feature_ranges == 0] = 1  # Avoid division by zero
            normalized_features = (features - features.min(axis=0)) / feature_ranges
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.2, min_samples=1).fit(normalized_features)
            labels = clustering.labels_
            
            # Keep the largest proposal from each cluster
            unique_labels = np.unique(labels)
            cluster_representatives = []
            
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_areas = filtered_proposals[cluster_indices][:, 2] * filtered_proposals[cluster_indices][:, 3]
                representative_idx = cluster_indices[np.argmax(cluster_areas)]
                cluster_representatives.append(filtered_proposals[representative_idx])
            
            filtered_proposals = np.array(cluster_representatives)
        
        # 5. Limit to max_proposals by taking the largest ones
        if len(filtered_proposals) > max_proposals:
            proposal_areas = filtered_proposals[:, 2] * filtered_proposals[:, 3]
            largest_indices = np.argsort(proposal_areas)[-max_proposals:]
            filtered_proposals = filtered_proposals[largest_indices]
        
        # Convert back to list of tuples
        return [tuple(map(int, proposal)) for proposal in filtered_proposals]
    
    def _custom_nms(self, boxes, iou_threshold=0.5):
        """
        Custom implementation of Non-Maximum Suppression.
        
        Args:
            boxes: Array of boxes in format [x1, y1, x2, y2, score]
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
            
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
        
        return keep
    
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
        self.ss.setBaseImage(img)
        self.ss.switchToSelectiveSearchFast()
        rects = self.ss.process()

        # Initial filter for tiny regions
        initial_filtered = []
        for rect in rects:
            x, y, w, h = rect
            if w >= 20 and h >= 20:  # Minimum size requirement
                initial_filtered.append(rect)
        
        print(f"Initial proposals: {len(rects)}, after basic filtering: {len(initial_filtered)}")
        
        # Apply advanced filtering
        filtered_rects = self.filter_proposals(img, initial_filtered, max_proposals)
        
        print(f"After advanced filtering: {len(filtered_rects)} proposals")
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
    ref_roi = region_generator.set_reference(reference_image, save_path="reference.jpg")
    
    test_image = "test_HVAC_batch/has_hvac_3.jpg"
    detections = region_generator.detect_objects(test_image, max_proposals=150)
    filtered_detections = region_generator.non_max_suppression(detections, iou_threshold=0.5)
    
    region_generator.visualize_detections(test_image, filtered_detections, output_path="detected_hvac.jpg")
    
    results = region_generator.batch_process("test_HVAC_batch", "output_detections")