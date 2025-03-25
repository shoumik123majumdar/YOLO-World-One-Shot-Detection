import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import sys
from ultralytics import YOLO 

def run_yolo_world_detection(
    model_path,
    test_dir, 
    output_dir, 
    has_hvac_filenames=None,
    confidence_threshold=0.5, 
    iou_threshold=0.5,
    device=None 
):
    """
    Run custom YOLO-World model for HVAC detection and evaluate performance.
    
    Args:
        model_path: Path to custom YOLO-World model
        test_dir: Directory containing test images
        output_dir: Directory to save results
        has_hvac_filenames: List of filenames known to contain HVAC units (ground truth)
        confidence_threshold: Minimum confidence score for detection
        iou_threshold: IoU threshold for NMS
        device: Device to run model on (None for auto-select)
    
    Returns:
        Dictionary with performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading custom YOLO-World model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Model loaded successfully")
    
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
            
            has_hvac_gt = False
            if has_hvac_filenames:
                has_hvac_gt = filename in has_hvac_filenames
            
            
            results = model.predict(
                image_path,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False 
            )
            
            result = results[0]
            
            detected_hvac = len(result.boxes) > 0
            best_score = float(result.boxes.conf[0]) if detected_hvac else 0
            
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
            
            
            result_img = result.plot()
            
            result_img = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
            
            
            h, w = result_img.shape[:2]
            overlay = result_img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)
            
            result_color = {
                "TP": (0, 255, 0),
                "TN": (255, 255, 255), 
                "FP": (0, 0, 255),   
                "FN": (0, 165, 255)  
            }
            result_text = f"Result: {result_type}"
            text_size, _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(result_img, result_text, (w - text_size[0] - 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, result_color[result_type], 2)
            
            cv2.imwrite(output_path, result_img)
            
            
            csv_writer.writerow([filename, has_hvac_gt, detected_hvac, best_score, result_type])
    
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("HVAC Detection Performance Summary\n")
        f.write("=================================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Directory: {test_dir}\n")
        f.write(f"Confidence Threshold: {confidence_threshold}\n")
        f.write(f"IoU Threshold: {iou_threshold}\n\n")
        
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
    model_path = "yolov8l-world.pt"  
    test_dir = "test_HVAC_batch"  
    output_dir = "custom_hvac_results" 
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        test_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
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
    
    metrics = run_yolo_world_detection(
        model_path=model_path,
        test_dir=test_dir,
        output_dir=output_dir,
        has_hvac_filenames=hvac_images,
        confidence_threshold=0.5, 
        iou_threshold=0.5
    )