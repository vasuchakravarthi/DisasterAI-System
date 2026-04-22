# app/gradcam.py - FIXED WITH GREEN BOUNDING BOXES AROUND CRACKS ONLY
import cv2
import numpy as np
import tensorflow as tf
import os

def get_last_conv_layer(model):
    """Find the last convolutional layer for Grad-CAM"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, layer
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name, sublayer
    return 'Conv_1', None

def compute_gradcam(model, img_array, layer_name=None):
    """Compute Grad-CAM heatmap for the image"""
    try:
        if layer_name is None:
            layer_name, _ = get_last_conv_layer(model)
        
        print(f"📌 Using Grad-CAM layer: {layer_name}")
        
        if isinstance(model, tf.keras.Sequential):
            base_model = model.layers[0]
            grad_model = tf.keras.models.Model(
                inputs=base_model.inputs,
                outputs=[base_model.get_layer(layer_name).output, model.output]
            )
        else:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer_name).output, model.outputs[0]]
            )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_output)
        
        if grads is None:
            print("⚠️ Gradients are None, using fallback")
            return None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        heatmap = np.maximum(heatmap.numpy(), 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
        
    except Exception as e:
        print(f"⚠️ Grad-CAM computation error: {e}")
        return None

def apply_colormap(heatmap, original_img):
    """Apply colormap to heatmap and overlay on image"""
    if heatmap is None:
        return None, None
    
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    return heatmap_colored, overlay

def get_crack_contours(image, model, tile_size=256, step=128):
    """
    Detect crack regions and return contours for precise bounding boxes
    """
    h, w = image.shape[:2]
    crack_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Step 1: Tile-based prediction to create crack probability map
    for y in range(0, h - tile_size, step):
        for x in range(0, w - tile_size, step):
            tile = image[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue
            
            tile_resized = cv2.resize(tile, (256, 256))
            tile_normalized = tile_resized.astype(np.float32) / 255.0
            tile_batch = np.expand_dims(tile_normalized, axis=0)
            prob = model.predict(tile_batch, verbose=0)[0][0]
            
            if prob > 0.5:
                # Mark this region as crack
                crack_mask[y:y+tile_size, x:x+tile_size] = 255
    
    # Step 2: Apply morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel)
    
    # Step 3: Find contours of crack regions
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Filter small contours (noise)
    min_area = 100  # Minimum crack area in pixels
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours, crack_mask

def generate_crack_visualization(model, image_path):
    """Generate Grad-CAM heatmap and GREEN bounding boxes around cracks only"""
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            return {"success": False, "error": "Cannot read image"}
        
        h, w = original_img.shape[:2]
        
        # Preprocess for model
        img_resized = cv2.resize(original_img, (256, 256))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Get prediction
        prediction = model.predict(img_batch, verbose=0)[0][0]
        crack_probability = float(prediction)
        
        print(f"Prediction: {crack_probability:.4f} - {'CRACK' if crack_probability > 0.5 else 'NO CRACK'}")
        
        # ============================================
        # 1. GET PRECISE CRACK CONTOURS
        # ============================================
        contours, crack_mask = get_crack_contours(original_img, model)
        
        # ============================================
        # 2. DRAW GREEN BOUNDING BOXES (ONLY AROUND ACTUAL CRACKS)
        # ============================================
        bbox_image = original_img.copy()
        bounding_boxes = []
        
        if len(contours) > 0:
            print(f"✅ Detected {len(contours)} crack region(s)")
            
            for i, contour in enumerate(contours):
                # Get bounding rectangle for each crack contour
                x, y, box_w, box_h = cv2.boundingRect(contour)
                
                # Add small padding (optional)
                padding = 3
                x = max(0, x - padding)
                y = max(0, y - padding)
                box_w = min(w - x, box_w + 2 * padding)
                box_h = min(h - y, box_h + 2 * padding)
                
                # Draw GREEN rectangle (BGR color = (0, 255, 0))
                cv2.rectangle(bbox_image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                
                # Add crack label with number
                label = f"Crack {i+1}"
                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(bbox_image, (x, y - label_h - 5), (x + label_w + 5, y), (0, 255, 0), -1)
                # Draw label text
                cv2.putText(bbox_image, label, (x + 2, y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                bounding_boxes.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(box_w),
                    "height": int(box_h),
                    "confidence": crack_probability,
                    "label": label
                })
            
            # Also draw green contours on the image (shows exact crack shape)
            cv2.drawContours(bbox_image, contours, -1, (0, 255, 0), 1)
            
        else:
            print("✅ No cracks detected - no bounding boxes drawn")
            # Add text indicating no cracks found
            cv2.putText(bbox_image, "NO CRACKS DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # ============================================
        # 3. GENERATE GRAD-CAM HEATMAP
        # ============================================
        heatmap = compute_gradcam(model, img_batch)
        
        heatmap_colored = None
        overlay_image = None
        
        if heatmap is not None:
            heatmap_colored, overlay_image = apply_colormap(heatmap, original_img)
        else:
            # Fallback: use crack mask as heatmap
            fallback_heatmap = crack_mask.astype(np.float32) / 255.0
            heatmap_colored, overlay_image = apply_colormap(fallback_heatmap, original_img)
        
        # ============================================
        # 4. CREATE CRACK-ONLY HEATMAP (Activation Map)
        # ============================================
        # This shows only where cracks are detected (red regions)
        crack_only_heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        if len(contours) > 0:
            for contour in contours:
                cv2.drawContours(crack_only_heatmap, [contour], -1, (0, 0, 255), -1)
        
        # ============================================
        # 5. SAVE IMAGES
        # ============================================
        os.makedirs("static", exist_ok=True)
        
        bbox_path = "static/bounding_boxes.jpg"
        overlay_path = "static/gradcam_overlay.jpg"
        heatmap_path = "static/gradcam_heatmap.jpg"
        
        cv2.imwrite(bbox_path, bbox_image)
        
        if overlay_image is not None:
            cv2.imwrite(overlay_path, overlay_image)
        else:
            cv2.imwrite(overlay_path, bbox_image)
        
        if heatmap_colored is not None:
            cv2.imwrite(heatmap_path, heatmap_colored)
        else:
            # Save crack-only visualization
            cv2.imwrite(heatmap_path, crack_only_heatmap)
        
        return {
            "success": True,
            "crack_probability": crack_probability,
            "bounding_boxes": bounding_boxes,
            "num_cracks": len(contours),
            "bbox_path": bbox_path,
            "overlay_path": overlay_path,
            "heatmap_path": heatmap_path,
            "confidence": crack_probability if crack_probability > 0.5 else 1 - crack_probability
        }
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "bounding_boxes": [],
            "crack_probability": 0,
            "num_cracks": 0,
            "bbox_path": None,
            "overlay_path": None,
            "heatmap_path": None
        }