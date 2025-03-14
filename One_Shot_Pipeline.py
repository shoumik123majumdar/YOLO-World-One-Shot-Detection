import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import time

# Load YOLO-World model
model = YOLO("yolov8-world.pt")

def selective_search(image, max_proposals=200):
    """
    Apply Selective Search to generate object proposals.
    
    Args:
        image: Input image (either a path or an image array)
        max_proposals: Maximum number of proposals to return
        
    Returns:
        List of proposal rectangles in format (x, y, w, h)
    """
    # Load image if path is provided
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
        
    # Initialize Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    
    # Use fast mode for real-time applications
    ss.switchToSelectiveSearchFast()
    
    # Generate proposals (returns a list of (x, y, w, h) rectangles)
    proposals = ss.process()[:max_proposals]
    
    print(f"Generated {len(proposals)} initial proposals")
    return proposals

def filter_boxes(proposals, image_shape, min_size=20, max_size_ratio=0.8, max_aspect_ratio=3.0):
    """
    Filter region proposals based on size and aspect ratio constraints.
    
    Args:
        proposals: List of proposal rectangles in format (x, y, w, h)
        image_shape: Tuple of (height, width) of the image
        min_size: Minimum width/height for a valid box. Used to filter out noise/unecessary texture
        max_size_ratio: Maximum size as a ratio of image dimensions. 
        max_aspect_ratio: Maximum allowed aspect ratio (w/h or h/w)
        
    Returns:
        Filtered list of proposals
    """
    img_height, img_width = image_shape[:2]
    max_width = int(img_width * max_size_ratio)
    max_height = int(img_height * max_size_ratio)
    
    filtered = []
    for x, y, w, h in proposals:
        # Skip tiny boxes
        if w < min_size or h < min_size:
            continue
            
        # Skip huge boxes
        if w > max_width or h > max_height:
            continue
            
        # Skip boxes with extreme aspect ratio
        aspect_ratio = max(w/h, h/w) if h > 0 else float('inf')
        if aspect_ratio > max_aspect_ratio:
            continue
            
        filtered.append((x, y, w, h))
    
    print(f"Kept {len(filtered)} proposals after filtering")
    return filtered

def extract_features(model, image, region=None):
    """
    Extract features from an image or a specific region using YOLO-World backbone.
    
    Args:
        model: YOLO model
        image: Input image
        region: Optional (x, y, w, h) region to extract features from
        
    Returns:
        Feature tensor
    """
    # Load image if path is provided
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    # Crop region if specified
    if region is not None:
        x, y, w, h = region
        img = img[y:y+h, x:x+w]
    
    # Ensure image is not empty
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    
    # Convert to RGB (YOLO expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract features - this is a simplified approach
    # In a real implementation, you'd access the backbone features more directly
    with torch.no_grad():
        results = model(img_rgb, verbose=False)
        # This is a placeholder - you'd need to adapt this to actually access
        # the backbone features properly in your implementation
        features = torch.tensor(np.mean(img_rgb, axis=(0, 1))).float()
        features = F.normalize(features, dim=0)
    
    return features

def match_regions(model, reference_features, proposals, image, similarity_threshold=0.7):
    """
    Compare one-shot reference features with proposed regions.
    
    Args:
        model: YOLO model
        reference_features: Features from reference image
        proposals: Filtered region proposals
        image: Test image
        similarity_threshold: Threshold for considering a match
        
    Returns:
        List of (box, score) tuples for matches above threshold
    """
    matches = []
    
    # Process each proposal
    for i, rect in enumerate(proposals):
        # Extract features from the proposed region
        region_features = extract_features(model, image, rect)
        
        # Skip if feature extraction failed
        if region_features is None:
            continue
        
        # Calculate similarity score
        similarity = F.cosine_similarity(
            reference_features.unsqueeze(0),
            region_features.unsqueeze(0)
        ).item()
        
        # Keep matches above threshold
        if similarity >= similarity_threshold:
            matches.append((rect, similarity))
    
    print(f"Found {len(matches)} matches above threshold {similarity_threshold}")
    return matches

def non_max_suppression(matches, iou_threshold=0.5):
    """
    Apply non-maximum suppression to avoid duplicate detections.
    
    Args:
        matches: List of (box, score) tuples
        iou_threshold: Boxes with IoU higher than this will be suppressed
        
    Returns:
        Filtered list of matches
    """
    if not matches:
        return []
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    
    keep = []
    
    while matches:
        # Keep highest scoring match
        best_match = matches.pop(0)
        keep.append(best_match)
        
        # Filter out overlapping detections
        remaining_matches = []
        best_rect = best_match[0]  # (x, y, w, h)
        
        for other_match in matches:
            other_rect = other_match[0]
            
            # Calculate IoU
            iou = calculate_iou(best_rect, other_rect)
            
            # Keep if IoU is below threshold
            if iou < iou_threshold:
                remaining_matches.append(other_match)
        
        matches = remaining_matches
    
    print(f"Kept {len(keep)} detections after NMS")
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (xmin, ymin, xmax, ymax) format
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def detect_object(model, reference_image, test_image, label, max_proposals=200, 
                  similarity_threshold=0.7, visualize=True):
    """
    Complete one-shot object detection pipeline.
    
    Args:
        model: YOLO model
        reference_image: Path or array of reference image
        test_image: Path or array of test image
        label: Label for the reference object
        max_proposals: Maximum number of initial proposals
        similarity_threshold: Threshold for considering a match
        visualize: Whether to visualize and save results
        
    Returns:
        List of final detections
    """
    # Timing for performance measurement
    start_time = time.time()
    
    # Step 1: Load images
    if isinstance(reference_image, str):
        ref_img = cv2.imread(reference_image)
    else:
        ref_img = reference_image
        s
    if isinstance(test_image, str):
        test_img = cv2.imread(test_image)
    else:
        test_img = test_image
        
    # Step 2: Extract reference features
    ref_features = extract_features(model, ref_img)
    feature_time = time.time() - start_time
    print(f"Reference feature extraction: {feature_time:.3f} seconds")
    
    # Step 3: Generate proposals
    proposal_start = time.time()
    proposals = selective_search(test_img, max_proposals)
    proposal_time = time.time() - proposal_start
    print(f"Proposal generation: {proposal_time:.3f} seconds")
    
    # Step 4: Filter proposals
    filtering_start = time.time()
    filtered_proposals = filter_boxes(proposals, test_img.shape)
    filtering_time = time.time() - filtering_start
    print(f"Proposal filtering: {filtering_time:.3f} seconds")
    
    # Step 5: Match with reference
    matching_start = time.time()
    matches = match_regions(model, ref_features, filtered_proposals, test_img, similarity_threshold)
    matching_time = time.time() - matching_start
    print(f"Feature matching: {matching_time:.3f} seconds")
    
    # Step 6: Apply NMS
    nms_start = time.time()
    final_detections = non_max_suppression(matches)
    nms_time = time.time() - nms_start
    print(f"NMS: {nms_time:.3f} seconds")
    
    # Total time
    total_time = time.time() - start_time
    print(f"Total detection time: {total_time:.3f} seconds ({1/total_time:.2f} FPS)")
    
    # Visualize results
    if visualize and final_detections:
        result_img = test_img.copy()
        
        # Draw final detections
        for (x, y, w, h), score in final_detections:
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Add label and score
            text = f"{label}: {score:.2f}"
            cv2.putText(result_img, text, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save result
        output_path = "one_shot_detection_result.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to {output_path}")
    
    return final_detections

# Example usage
if __name__ == "__main__":
    # Paths to your images
    reference_path = "reference_mug.jpg"
    test_path = "kitchen_scene.jpg"
    
    # Run detection
    detections = detect_object(
        model=model,
        reference_image=reference_path,
        test_image=test_path,
        label="coffee mug",
        max_proposals=200,
        similarity_threshold=0.7,
        visualize=True
    )