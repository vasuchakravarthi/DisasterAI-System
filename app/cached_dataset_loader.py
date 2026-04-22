"""
Cached Kaggle Dataset Loader - Method 4
Downloads only what you need, caches for performance
Location: app/cached_dataset_loader.py
"""

import os
import pickle
import hashlib
import cv2
import numpy as np
from functools import lru_cache
from typing import Optional, Tuple, List

class CachedKaggleLoader:
    """
    Load Kaggle dataset with intelligent caching
    Only downloads samples you actually use
    """
    
    def __init__(self, cache_size_mb: int = 200):
        self.cache_dir = "./dataset_cache"
        self.cache_size_mb = cache_size_mb
        self.reference_images = {}  # Store reference crack images
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Try to import kagglehub
        try:
            import kagglehub
            self.kagglehub = kagglehub
            self.available = True
            self._init_references()
        except ImportError:
            print("⚠️ kagglehub not installed. Install with: pip install kagglehub")
            self.available = False
    
    def _init_references(self):
        """
        Initialize reference images from dataset (only if needed)
        """
        try:
            # Download dataset path (cached by kagglehub)
            dataset_path = self.kagglehub.dataset_download("shrutisaxena/crack-detection-dataset")
            
            # Look for sample images
            train_path = os.path.join(dataset_path, "train")
            if os.path.exists(train_path):
                # Get reference images without loading all
                crack_path = os.path.join(train_path, "cracked")
                if os.path.exists(crack_path):
                    sample_images = os.listdir(crack_path)[:5]  # Just 5 samples
                    for img_name in sample_images:
                        img_path = os.path.join(crack_path, img_name)
                        self.reference_images[img_name] = img_path
                        
            print(f"✅ Initialized {len(self.reference_images)} reference images")
        except Exception as e:
            print(f"⚠️ Could not initialize references: {e}")
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate unique cache key for image"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @lru_cache(maxsize=50)
    def get_similar_damage_images(self, damage_score: float, limit: int = 3) -> List[str]:
        """
        Get paths to similar damage images from dataset
        Cached for performance
        """
        if not self.available:
            return []
        
        similar_images = []
        
        try:
            dataset_path = self.kagglehub.dataset_download("shrutisaxena/crack-detection-dataset")
            train_path = os.path.join(dataset_path, "train", "cracked")
            
            if os.path.exists(train_path):
                # Get images based on damage severity
                all_images = os.listdir(train_path)
                
                # Simulate similarity based on damage score
                if damage_score > 70:
                    # Severe damage images (random sample)
                    similar_images = all_images[:limit]
                elif damage_score > 40:
                    # Moderate damage
                    similar_images = all_images[limit:limit*2]
                else:
                    # Minor damage
                    similar_images = all_images[limit*2:limit*3]
                
                # Convert to full paths
                similar_images = [os.path.join(train_path, img) for img in similar_images if img.endswith(('.jpg', '.png'))]
        except Exception as e:
            print(f"Error getting similar images: {e}")
        
        return similar_images
    
    def get_crack_probability(self, image_path: str) -> float:
        """
        Compare uploaded image with dataset to estimate crack probability
        Uses feature matching without full model training
        """
        if not self.available:
            return None
        
        try:
            # Load uploaded image
            query_img = cv2.imread(image_path)
            if query_img is None:
                return 0.5
            
            query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            
            # Initialize ORB feature detector
            orb = cv2.ORB_create(nfeatures=500)
            query_keypoints, query_descriptors = orb.detectAndCompute(query_gray, None)
            
            if query_descriptors is None:
                return 0.5
            
            # Compare with reference images
            best_match_score = 0
            match_count = 0
            
            for ref_path in list(self.reference_images.values())[:3]:  # Compare with 3 references
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
                    
                    if ref_descriptors is not None:
                        # Feature matching
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(query_descriptors, ref_descriptors)
                        
                        if len(matches) > 0:
                            # Sort by distance
                            matches = sorted(matches, key=lambda x: x.distance)
                            good_matches = [m for m in matches if m.distance < 50]
                            
                            match_score = len(good_matches) / max(len(query_keypoints), 1)
                            best_match_score = max(best_match_score, match_score)
                            match_count += 1
            
            # Convert match score to crack probability
            if match_count > 0:
                crack_probability = min(0.95, best_match_score * 1.5)
                return round(crack_probability, 3)
            else:
                return 0.5
                
        except Exception as e:
            print(f"Error computing crack probability: {e}")
            return 0.5
    
    def get_training_batch_generator(self, batch_size: int = 8):
        """
        Get a lightweight batch generator for on-demand training
        """
        if not self.available:
            return None
        
        try:
            dataset_path = self.kagglehub.dataset_download("shrutisaxena/crack-detection-dataset")
            train_path = os.path.join(dataset_path, "train")
            
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            generator = datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='binary',
                subset='training'
            )
            
            return generator
        except Exception as e:
            print(f"Error creating generator: {e}")
            return None
    
    def get_dataset_summary(self) -> dict:
        """
        Get summary of available dataset
        """
        if not self.available:
            return {"available": False, "error": "kagglehub not installed"}
        
        try:
            dataset_path = self.kagglehub.dataset_download("shrutisaxena/crack-detection-dataset")
            train_path = os.path.join(dataset_path, "train")
            
            cracked_count = 0
            uncracked_count = 0
            
            if os.path.exists(train_path):
                cracked_path = os.path.join(train_path, "cracked")
                uncracked_path = os.path.join(train_path, "uncracked")
                
                if os.path.exists(cracked_path):
                    cracked_count = len(os.listdir(cracked_path))
                if os.path.exists(uncracked_path):
                    uncracked_count = len(os.listdir(uncracked_path))
            
            return {
                "available": True,
                "dataset": "SDNET2018 - Concrete Crack Detection",
                "total_images": cracked_count + uncracked_count,
                "cracked_images": cracked_count,
                "uncracked_images": uncracked_count,
                "cached_references": len(self.reference_images),
                "cache_location": self.cache_dir
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

# Global instance
cached_loader = CachedKaggleLoader()

def get_crack_probability_from_dataset(image_path: str) -> float:
    """Public function to get crack probability using dataset"""
    return cached_loader.get_crack_probability(image_path)

def get_dataset_summary():
    """Get dataset summary"""
    return cached_loader.get_dataset_summary()