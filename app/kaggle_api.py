"""
Direct Kaggle API Integration - Method 3
Access dataset metadata and files without downloading
Location: app/kaggle_api.py
"""

import requests
import json
import os
from typing import Dict, List, Optional

class DirectKaggleAPI:
    """
    Direct Kaggle API access without dataset download
    """
    
    def __init__(self):
        self.base_url = "https://www.kaggle.com/api/v1"
        self.datasets = {
            "crack_detection": "shrutisaxena/crack-detection-dataset",
            "bridge_defects": "sulaimanmughal/bridge-defect-detection",
            "concrete_crack": "arunrk7/surface-crack-detection"
        }
    
    def get_dataset_info(self, dataset_name: str = "crack_detection") -> Dict:
        """
        Get dataset metadata without downloading
        """
        dataset_path = self.datasets.get(dataset_name, self.datasets["crack_detection"])
        url = f"{self.base_url}/datasets/{dataset_path}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned {response.status_code}", "dataset": dataset_path}
        except Exception as e:
            return {"error": str(e)}
    
    def get_dataset_files(self, dataset_name: str = "crack_detection") -> List[Dict]:
        """
        List files in dataset without downloading
        """
        dataset_path = self.datasets.get(dataset_name, self.datasets["crack_detection"])
        url = f"{self.base_url}/datasets/{dataset_path}/files"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            print(f"Error fetching files: {e}")
            return []
    
    def get_dataset_stats(self, dataset_name: str = "crack_detection") -> Dict:
        """
        Get dataset statistics (size, number of images, classes)
        """
        info = self.get_dataset_info(dataset_name)
        files = self.get_dataset_files(dataset_name)
        
        return {
            "name": info.get("title", dataset_name),
            "description": info.get("description", "")[:200],
            "file_count": len(files),
            "total_size_mb": sum(f.get("size", 0) for f in files) / (1024 * 1024),
            "owner": info.get("ownerName", "Unknown"),
            "last_updated": info.get("lastUpdated", "Unknown")
        }
    
    def search_similar_images(self, damage_score: float) -> List[Dict]:
        """
        Search for similar damage patterns in dataset (metadata only)
        """
        # This queries Kaggle's metadata for similar classifications
        results = []
        
        # Based on damage score, find relevant dataset entries
        if damage_score > 70:
            severity = "severe"
            search_term = "severe crack structural damage"
        elif damage_score > 40:
            severity = "moderate"
            search_term = "moderate crack concrete"
        else:
            severity = "minor"
            search_term = "hairline crack surface"
        
        # Search Kaggle datasets (API call)
        search_url = f"{self.base_url}/datasets/list?search={search_term}"
        
        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                results = response.json()[:5]  # Top 5 matches
        except:
            pass
        
        return results
    
    def get_crack_classification_examples(self) -> Dict:
        """
        Get classification examples from dataset
        """
        return {
            "crack_types": [
                {"type": "Hairline Crack", "severity": "Low", "repair": "Surface sealant"},
                {"type": "Structural Crack", "severity": "High", "repair": "Epoxy injection"},
                {"type": "Spalling", "severity": "Medium", "repair": "Concrete patch"},
                {"type": "Exposed Rebar", "severity": "Critical", "repair": "Full structural repair"}
            ],
            "dataset_source": self.datasets["crack_detection"],
            "total_images": "56,000+",
            "classes": ["Cracked", "Uncracked"]
        }

# Create global instance
kaggle_api = DirectKaggleAPI()

def get_dataset_info():
    """Public function to get dataset info"""
    return kaggle_api.get_dataset_stats()

def get_crack_examples():
    """Get crack classification examples"""
    return kaggle_api.get_crack_classification_examples()
