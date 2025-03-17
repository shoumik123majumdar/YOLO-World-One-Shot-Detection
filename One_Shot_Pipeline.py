import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np

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

def extract_features(model, image_path):
    """
    Extract features from an image using a specific YOLO layer.
    
    Args:
        model: YOLO model
        image_path: Path to image or image array
        embed_layer: Layer index to extract embeddings from (default: 10)
    
    Returns:
        Embedding tensor or None if extraction failed
    """
    # Run prediction with embedding extraction enabled
    results = model.predict(source=image_path, embed=[10],verbose=False)
    for result in results:
        embeddings = result.embeddings
    return results


def match_regions(model, reference_image, proposals, test_image, similarity_threshold=0.7):
    """
    Compare one-shot reference features with proposed regions.
    
    Args:
        model: YOLO model
        reference_image: Path to reference image
        proposals: List of region proposals [x, y, w, h]
        test_image: Path to test image
        similarity_threshold: Minimum similarity to be considered a match
        
    Returns:
        (best_match, best_score) tuple or None if no good match found
    """
    # Extract features from reference image
    reference_features = extract_features(model, reference_image)
    
    img = cv2.imread(test_image)
    
    best_match = None
    best_score = -1
    # Process each region proposal
    for i, rect in enumerate(proposals):
        try:
            x, y, w, h = rect
            
            # Skip invalid regions
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                continue

            crop = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Extract features from the crop
            query_features = extract_features(model, crop_rgb)
            
            if query_features is not None:
                # Calculate cosine similarity
                similarity = F.cosine_similarity(reference_features, query_features, dim=1).item()
                
                # Update best match if this is better
                if similarity > best_score:
                    best_score = similarity
                    best_match = rect
                print(f"Region {i+1}: Similarity = {similarity:.4f}", end="\r")
                
        except Exception as e:
            print(f"Error processing region {i+1}: {e}")
            continue
    
    print(f"\nBest match similarity: {best_score:.4f}")
    # Return the best match if it's above the threshold
    if best_score >= similarity_threshold:
        return (best_match, best_score)
    else:
        return None

def visualize_match(image_path, match_rect, score, output_path=None):
    """
    Visualize the best matching region on the image.
    
    Args:
        image_path: Path to the image
        match_rect: Rectangle coordinates [x, y, w, h]
        score: Similarity score
        output_path: Path to save visualization (if None, displays on screen)
        
    Returns:
        Image with visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    img_copy = img.copy()
    
    # Draw the matching rectangle
    x, y, w, h = match_rect
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # Add similarity score text
    text = f"Similarity: {score:.2f}"
    cv2.putText(img_copy, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display or save the result
    if output_path:
        cv2.imwrite(output_path, img_copy)
        print(f"Match visualization saved to {output_path}")
    else:
        cv2.imshow("Best Match", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return img_copy

# Main execution
if __name__ == "__main__":
    # Load YOLO model
    model_path = "yolov8n.pt"  # Change to your model path
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    

    # Paths to images
    reference_image = "One_Shot_HVAC.jpg"
    test_image = "test_image_1.jpg"

    #Embedding extraction test
    print(f"with feature layer param: {extract_features(model,reference_image)}")
    print(f"without: {extract_featuress(model,reference_image)}")

    
    # Generate region proposals
    print(f"Generating region proposals for {test_image}")
    proposals = selective_search(test_image, max_proposals=50)
    print(f"Generated {len(proposals)} region proposals")
    
    # Match regions
    print(f"Comparing reference image to region proposals...")
    match_result = match_regions(model, reference_image, proposals, test_image)
    
    # Visualize results
    if match_result:
        best_match, best_score = match_result
        print(f"Best match found with similarity: {best_score:.4f}")
        visualize_match(test_image, best_match, best_score, output_path="best_match.jpg")
    else:
        print("No suitable match found")