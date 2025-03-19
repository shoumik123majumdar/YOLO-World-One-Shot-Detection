import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import os
import csv
from tqdm import tqdm

"""
10 with HVAC
10 without HVAC
See how many false positives/negatives does it too
Test accuracy without one shot
ALSO DISPLAY THE LABEL NAME OUTSIDE THE BOUNDING BOX
IF multiple of the reference object, also show the next 3 greatest proposals or something
"""


def selective_search(image_path, max_proposals=50):
    """Apply Selective Search to generate object proposals
    Args:
        image_path (str): Path to the input image
        max_proposals (int,optional): Maximum number of proposals to be generated
    """
    img = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:max_proposals] # Get Top 50 Object Proposals
    return rects

def show_region_proposals(image_path, rects):
    """
    Draw region proposals (bounding boxes) on the image and display them.
    
    Args:
        image_path (str): Path to the input image
        rects (list): List of regions/rectangles in format (x, y, w, h)
    
    Returns:
        numpy.ndarray: The image with bounding boxes drawn
    """
    img = cv2.imread(image_path)
    img_copy = img.copy()
    red_color = (0, 0, 255)

    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), red_color, 2)
    
    cv2.imshow("Region Proposals", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img_copy

def extract_features(model, image_path, layer_name="model.model.21.cv2", print_layers=False):
    # Print all available layer names if requested
    if print_layers:
        print("Available layers in the model:")
        for name, module in model.named_modules():
            # Print layer name and its type
            print(f"Layer: {name} - Type: {type(module).__name__}")
        print("-" * 50)  # Separator for readability
    
    # Store embeddings globally
    embeddings_output = None
    
    def hook_fn(module, input, output):
        nonlocal embeddings_output
        embeddings_output = output
    
    # Register hook on the specified layer
    try:
        layer = dict(model.named_modules())[layer_name]
        hook = layer.register_forward_hook(hook_fn)
        
        # Run inference to trigger the hook
        # Add 'verbose=False' to suppress unnecessary output from model
        model(image_path, verbose=False)
        
        # Remove the hook
        hook.remove()
        
        # Process embeddings if needed
        if embeddings_output is not None:
            # If embeddings have spatial dimensions (NCHW), pool them to a vector
            if embeddings_output.dim() > 2:
                # For layer 18.cv2, we're likely dealing with feature maps
                # that need to be pooled to a 1D vector
                embeddings = F.adaptive_avg_pool2d(embeddings_output, 1).squeeze(-1).squeeze(-1)
                
                # Normalize the feature vector for better similarity comparison
                embeddings = F.normalize(embeddings, p=2, dim=0)
                return embeddings
            return embeddings_output
            
        raise ValueError(f"Failed to extract embeddings from layer {layer_name}")
    
    except KeyError:
        print(f"Layer '{layer_name}' not found in the model.")
        print("Please choose from the available layers listed above.")
        return None


def match_regions(model, reference_image, proposals, test_image, top_n=3, similarity_threshold=0.7):
    """
    Compare one-shot reference features with proposed regions.
    
    Args:
        model: YOLO model
        reference_image: Path to reference image
        proposals: List of region proposals [x, y, w, h]
        test_image: Path to test image
        top_n: Number of top matches to return
        similarity_threshold: Minimum similarity to be considered a match
        
    Returns:
        List of (match_rect, score) tuples, sorted by score (descending)
    """
    # Extract features from reference image
    reference_features = extract_features(model, reference_image)
    
    img = cv2.imread(test_image)
    if img is None:
        print(f"Error: Could not read image at {test_image}")
        return []
    
    all_matches = []
    
    # Process each region proposal
    for i, rect in enumerate(proposals):
        try:
            x, y, w, h = rect
            
            # Skip invalid regions
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                continue

            crop = img[y:y+h, x:x+w]
            
            # Convert to RGB for the model
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Extract features from the crop
            query_features = extract_features(model, crop_rgb)
            
            if query_features is not None:
                # Calculate cosine similarity
                similarity = F.cosine_similarity(reference_features, query_features, dim=1).item()
                
                # Add to matches if above threshold
                if similarity >= similarity_threshold:
                    all_matches.append((rect, similarity))
                
        except Exception as e:
            print(f"Error processing region {i+1}: {e}")
            continue
    
    # Sort matches by similarity score (descending)
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N matches
    return all_matches[:top_n]

def visualize_matches(image_path, matches, output_path=None, label="HVAC"):
    """
    Visualize the best matching regions on the image.
    
    Args:
        image_path: Path to the image
        matches: List of (match_rect, score) tuples
        output_path: Path to save visualization
        label: Label to display for matches
        
    Returns:
        Image with visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    img_copy = img.copy()
    
    # Colors for different matches (BGR format)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    # Draw each matching rectangle
    for i, (match_rect, score) in enumerate(matches):
        color = colors[i % len(colors)]
        x, y, w, h = match_rect
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 3)
        
        # Add label and similarity score text OUTSIDE the box
        text = f"{label}: {score:.2f}"
        # Position text above the box
        cv2.putText(img_copy, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save the result
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_copy)
    
    return img_copy

def test_hvac_detection(model, reference_image, test_dir, output_dir, 
                       has_hvac_filenames=None, similarity_threshold=0.7, 
                       max_proposals=50, top_matches=3):
    """
    Test HVAC detection on multiple images and evaluate performance.
    
    Args:
        model: YOLO model
        reference_image: Path to reference image
        test_dir: Directory containing test images
        output_dir: Directory to save results
        has_hvac_filenames: List of filenames known to contain HVAC units (ground truth)
        similarity_threshold: Minimum similarity to be considered a match
        max_proposals: Maximum number of region proposals to generate
        top_matches: Number of top matches to visualize
        
    Returns:
        Dictionary with performance metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a CSV file to store results
    results_csv = os.path.join(output_dir, "detection_results.csv")
    
    # Initialize performance counters
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    all_scores = []
    
    # Prepare CSV file
    with open(results_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image", "Has_HVAC_GT", "Detection", "Best_Score", "Result"])
        
        # Process each image in the test directory
        test_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(test_files, desc="Processing images"):
            image_path = os.path.join(test_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Determine ground truth (whether image has HVAC)
            has_hvac_gt = False
            if has_hvac_filenames:
                has_hvac_gt = filename in has_hvac_filenames
            
            # Generate region proposals
            proposals = selective_search(image_path, max_proposals=max_proposals)
            
            # Match regions
            matches = match_regions(model, reference_image, proposals, image_path, 
                                    top_n=top_matches, similarity_threshold=similarity_threshold)
            
            # Evaluate detection
            detection_result = len(matches) > 0
            best_score = matches[0][1] if matches else 0
            all_scores.append(best_score)
            
            # Update performance counters
            if has_hvac_gt and detection_result:
                result = "TP"  # True Positive
                true_positives += 1
            elif not has_hvac_gt and not detection_result:
                result = "TN"  # True Negative
                true_negatives += 1
            elif not has_hvac_gt and detection_result:
                result = "FP"  # False Positive
                false_positives += 1
            else:  # has_hvac_gt and not detection_result
                result = "FN"  # False Negative
                false_negatives += 1
            
            # Visualize and save results
            if matches:
                visualize_matches(image_path, matches, output_path)
            
            # Write results to CSV
            csv_writer.writerow([filename, has_hvac_gt, detection_result, best_score, result])
    
    # Calculate performance metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Generate summary report
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("HVAC Detection Performance Summary\n")
        f.write("=================================\n\n")
        f.write(f"Reference Image: {reference_image}\n")
        f.write(f"Test Directory: {test_dir}\n")
        f.write(f"Similarity Threshold: {similarity_threshold}\n")
        f.write(f"Max Region Proposals: {max_proposals}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  True Positives: {true_positives}\n")
        f.write(f"  True Negatives: {true_negatives}\n")
        f.write(f"  False Positives: {false_positives}\n")
        f.write(f"  False Negatives: {false_negatives}\n\n")
        
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"  F1 Score: {f1_score:.4f}\n")
    
    # Plot the score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=20, alpha=0.7)
    plt.axvline(x=similarity_threshold, color='r', linestyle='--', label=f'Threshold ({similarity_threshold})')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "score_distribution.png"))
    
    # Return performance metrics
    metrics = {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    print(f"Detection results saved to {output_dir}")
    return metrics

# Main execution
if __name__ == "__main__":
    # Load YOLO model
    model_path = "yolov8n.pt"  # Change to your model path
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Paths to images and directories
    reference_image = "reference_object.jpg"
    test_dir = "test_HVAC_batch"
    output_dir = "test_HVAC_results"
    
    # List of images known to contain HVAC units (for ground truth evaluation)
    # Fill this list with filenames (without path) that contain HVAC units
    hvac_images = [
    "has_hvac_1.jpg",
    "has_hvac_2.jpg",
    "has_hvac_3.jpg",
    "has_hvac_4.jpg",
    "has_hvac_5.jpg",
    "has_hvac_6.jpg",
    "has_hvac_7.jpg",
    "has_hvac_8.jpg",
    "has_hvac_9.jpg",
    "has_hvac_10.jpg"
    ]

    
    # Run the test
    metrics = test_hvac_detection(
        model=model,
        reference_image=reference_image,
        test_dir=test_dir,
        output_dir=output_dir,
        has_hvac_filenames=hvac_images,
        similarity_threshold=0.7,  # Adjust as needed
        max_proposals=50,
        top_matches=3
    )
    
    # Print summary of results
    print("\nPerformance Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")