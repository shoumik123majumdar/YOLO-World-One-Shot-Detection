def evaluate_complete_pipeline(reference_images, test_data, similarity_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """Complete evaluation of one-shot detection pipeline"""
    
    results = {}
    
    # For each reference object
    for obj_name, ref_img_path in reference_images.items():
        ref_img = cv2.imread(ref_img_path)
        ref_features = extract_features_properly(model, ref_img)
        
        obj_results = {}
        
        # For each similarity threshold
        for threshold in similarity_thresholds:
            threshold_results = []
            
            # For each test image with this object
            for test_img_path, gt_boxes in test_data[obj_name]:
                test_img = cv2.imread(test_img_path)
                
                # Time the inference
                start_time = time.time()
                
                # Run detection pipeline
                proposals = selective_search(test_img)
                filtered_proposals = filter_boxes(proposals)
                
                detections = []
                for rect in filtered_proposals:
                    x, y, w, h = rect
                    crop = test_img[y:y+h, x:x+w]
                    query_features = extract_features_properly(model, crop)
                    similarity = F.cosine_similarity(ref_features, query_features, dim=-1)
                    if similarity > threshold:
                        detections.append((rect, similarity.item()))
                
                final_detections = non_max_suppression(detections)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Calculate detection metrics
                metrics = calculate_metrics(final_detections, gt_boxes)
                
                # Evaluate proposal quality
                proposal_metrics = evaluate_proposal_effectiveness(filtered_proposals, gt_boxes)
                
                # Combine metrics
                result = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "proposal_recall": proposal_metrics["proposal_recall"],
                    "inference_time": inference_time
                }
                
                threshold_results.append(result)
            
            # Average metrics across all test images at this threshold
            avg_results = average_metrics(threshold_results)
            obj_results[threshold] = avg_results
        
        # Calculate AP for this object
        precisions = [obj_results[t]["precision"] for t in similarity_thresholds]
        recalls = [obj_results[t]["recall"] for t in similarity_thresholds]
        ap = calculate_ap(np.array(precisions), np.array(recalls))
        
        # Store results for this object
        results[obj_name] = {
            "per_threshold": obj_results,
            "ap": ap,
            "best_threshold": find_best_threshold(obj_results)
        }
    
    # Calculate mAP
    map_score = np.mean([results[obj]["ap"] for obj in results])
    
    # Calculate average FPS
    avg_fps = 1.0 / np.mean([results[obj]["per_threshold"][0.7]["inference_time"] 
                            for obj in results])
    
    return {
        "per_object": results,
        "mAP": map_score,
        "avg_fps": avg_fps
    }