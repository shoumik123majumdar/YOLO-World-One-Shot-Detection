import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import sys
from CLIP_Embedding_Extractor import EmbeddingExtractor
from Region_Proposal_Filtered_Generator import RegionProposalGenerator
def run_embedding_based_detection(
    reference_image_path,
    reference_roi=None,
    test_dir=None,
    output_dir=None,
    has_hvac_filenames=None,
    similarity_threshold=0.83,
    iou_threshold=0.5,
    max_proposals=100,
    clip_model_path='openai/clip-vit-base-patch32'
):
    """
    Run embedding-based HVAC detection using CLIP and region proposals.
    
    Args:
        reference_image_path: Path to reference image containing an HVAC unit
        reference_roi: ROI of HVAC in reference image as (x,y,w,h), if None user will select
        test_dir: Directory containing test images
        output_dir: Directory to save results
        has_hvac_filenames: List of filenames known to contain HVAC units (ground truth)
        similarity_threshold: Minimum similarity score for detection
        iou_threshold: IoU threshold for NMS
        max_proposals: Maximum number of region proposals to evaluate per image
        clip_model_path: Path to CLIP model
    
    Returns:
        Dictionary with performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing CLIP-based embedding extractor with {clip_model_path}...")
    extractor = EmbeddingExtractor(clip_model_path=clip_model_path)
    
    print(f"Initializing region proposal generator with similarity threshold {similarity_threshold}...")
    region_generator = RegionProposalGenerator(
        embedding_extractor=extractor,
        similarity_threshold=similarity_threshold
    )
    
    # Set reference embedding
    print(f"Setting reference embedding from {reference_image_path}...")
    if reference_roi is None:
        print("Please select the HVAC unit in the reference image...")
        reference_roi = region_generator.set_reference(reference_image_path)
    else:
        region_generator.set_reference(reference_image_path, reference_roi)
    
    results_csv = os.path.join(output_dir, "detection_results.csv")
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with open(results_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image", "Has_HVAC_GT", "Detection", "Best_Score", "Result"])
        
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(test_files, desc="Processing images"):
            image_path = os.path.join(test_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Determine ground truth
            has_hvac_gt = False
            if has_hvac_filenames:
                has_hvac_gt = filename in has_hvac_filenames
            
            # Detect HVAC using region proposals and embedding comparison
            detections = region_generator.detect_objects(image_path, max_proposals)
            filtered_detections = region_generator.non_max_suppression(detections, iou_threshold)
            
            # Determine if HVAC was detected
            detected_hvac = len(filtered_detections) > 0
            best_score = filtered_detections[0][4] if detected_hvac else 0
            
            # Calculate metrics
            if has_hvac_gt and detected_hvac:
                result_type = "TP" 
                true_positives += 1
            elif not has_hvac_gt and not detected_hvac:
                result_type = "TN" 
                true_negatives += 1
            elif not has_hvac_gt and detected_hvac:
                result_type = "FP"  
                false_positives += 1
            else: 
                result_type = "FN"  
                false_negatives += 1
            
            # Save visualization with result
            vis_img = region_generator.visualize_detections(
                image_path, filtered_detections, show=False
            )
            
            # Convert back to BGR for OpenCV
            vis_img = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)
            
            # Add result text
            h, w = vis_img.shape[:2]
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis_img, 0.3, 0, vis_img)
            
            # Visualize detections and save without result text
            vis_img = region_generator.visualize_detections(
            image_path, filtered_detections, show=False
            )

            # Convert from RGB (from matplotlib) to BGR (for OpenCV)
            vis_img = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)

            # Direct save without adding overlay text
            cv2.imwrite(output_path, vis_img)

            # Write to CSV
            csv_writer.writerow([filename, has_hvac_gt, detected_hvac, best_score, result_type])
            
    
    # Calculate final metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Write summary
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("HVAC Detection Performance Summary (Embedding-Based)\n")
        f.write("=================================================\n\n")
        f.write(f"Reference Image: {reference_image_path}\n")
        f.write(f"Reference ROI: {reference_roi}\n")
        f.write(f"Test Directory: {test_dir}\n")
        f.write(f"CLIP Model: {clip_model_path}\n")
        f.write(f"Similarity Threshold: {similarity_threshold}\n")
        f.write(f"IoU Threshold: {iou_threshold}\n")
        f.write(f"Max Proposals: {max_proposals}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  True Positives: {true_positives}\n")
        f.write(f"  True Negatives: {true_negatives}\n")
        f.write(f"  False Positives: {false_positives}\n")
        f.write(f"  False Negatives: {false_negatives}\n\n")
        
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"  F1 Score: {f1_score:.4f}\n")
    
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
    print("\nPerformance Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    
    return metrics

if __name__ == "__main__":
    reference_image = "sample_1_reference.png"  # Default reference image
    test_dir = "sample2"  # Default test directory
    output_dir = "drone_hvac_results_run3_0.85"  # Default output directory
    
    # List of images known to contain HVAC units (ground truth)
    hvac_images = [f"{i}.png" for i in range(1, 27)]
    
    # Run the embedding-based detection
    metrics = run_embedding_based_detection(
        reference_image_path=reference_image,
        test_dir=test_dir,
        output_dir=output_dir,
        has_hvac_filenames=hvac_images,
        similarity_threshold=0.85,  # Default threshold from RegionProposalGenerator
        iou_threshold=0.5,
        max_proposals=200
    )