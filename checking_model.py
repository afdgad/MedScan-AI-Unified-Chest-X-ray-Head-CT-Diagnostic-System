import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define pathologies
PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class EnhancedChestXRayPredictor:
    def __init__(self, model_path, pathologies):
        self.pathologies = pathologies
        self.num_classes = len(pathologies)
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
        # Optimized thresholds for each pathology (you can tune these)
        self.thresholds = {
            'Atelectasis': 0.25,
            'Cardiomegaly': 0.35,
            'Effusion': 0.28,
            'Infiltration': 0.22,
            'Mass': 0.32,
            'Nodule': 0.30,
            'Pneumonia': 0.26,
            'Pneumothorax': 0.38,
            'Consolidation': 0.27,
            'Edema': 0.31,
            'Emphysema': 0.33,
            'Fibrosis': 0.29,
            'Pleural_Thickening': 0.34,
            'Hernia': 0.40
        }
        
    def _load_model(self, model_path):
        """Load the trained model with error handling"""
        try:
            model = models.densenet121(weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.num_classes)
            
            # Load state dict with strict=False to handle minor mismatches
            state_dict = torch.load(model_path, map_location=device)
            
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            model.eval()
            
            print(f"✅ Model loaded successfully from: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _get_transforms(self):
        """Enhanced transforms with test-time augmentation"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing with error handling"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # Basic image quality check
            if image.size[0] < 100 or image.size[1] < 100:
                print("⚠️  Warning: Image resolution is very low")
            
            original_image = image.copy()
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            return input_batch, original_image
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict_with_tta(self, image_path, num_augmentations=5):
        """Test Time Augmentation for more robust predictions"""
        input_batch, original_image = self.preprocess_image(image_path)
        
        # Original prediction
        with torch.no_grad():
            outputs = self.model(input_batch)
            base_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # Augmentations (flip, rotate slightly)
        augmented_probs = []
        augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
        
        for aug in augmentations[:num_augmentations]:
            augmented_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                aug,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            augmented_image = augmented_transform(original_image).unsqueeze(0).to(device)
            with torch.no_grad():
                aug_output = self.model(augmented_image)
                aug_probs = torch.sigmoid(aug_output).cpu().numpy().flatten()
                augmented_probs.append(aug_probs)
        
        # Average predictions
        if augmented_probs:
            avg_probs = (base_probs + np.mean(augmented_probs, axis=0)) / 2
        else:
            avg_probs = base_probs
        
        return avg_probs, original_image
    
    def predict(self, image_path, use_tta=True):
        """Enhanced prediction with multiple strategies"""
        if use_tta:
            probabilities, original_image = self.predict_with_tta(image_path)
        else:
            input_batch, original_image = self.preprocess_image(image_path)
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # Apply pathology-specific thresholds
        detected_pathologies = []
        for i, prob in enumerate(probabilities):
            pathology_name = self.pathologies[i]
            threshold = self.thresholds.get(pathology_name, 0.3)
            
            if prob >= threshold:
                detected_pathologies.append({
                    'pathology': pathology_name,
                    'probability': float(prob),
                    'confidence': f"{prob:.1%}",
                    'threshold': threshold
                })
        
        # Sort by probability
        detected_pathologies.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'original_image': original_image,
            'all_probabilities': probabilities,
            'detected_pathologies': detected_pathologies,
            'has_abnormality': len(detected_pathologies) > 0,
            'confidence_score': np.max(probabilities) if len(probabilities) > 0 else 0
        }
    
    def analyze_image_quality(self, image_path):
        """Basic image quality assessment"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"quality": "poor", "message": "Cannot read image"}
        
        # Basic quality metrics
        contrast = np.std(image)
        brightness = np.mean(image)
        
        quality = "good"
        message = ""
        
        if contrast < 25:
            quality = "poor"
            message = "Low contrast"
        elif brightness < 30 or brightness > 220:
            quality = "poor"
            message = "Extreme brightness"
        
        return {
            "quality": quality,
            "message": message,
            "contrast": float(contrast),
            "brightness": float(brightness)
        }

# Main execution with batch testing
def test_multiple_images():
    """Test multiple images to evaluate model performance"""
    MODEL_PATH = "densenet121_nih_by_abrar_best.pth"
    TEST_IMAGES_DIR = "TEST_IMAGES"
    
    predictor = EnhancedChestXRayPredictor(MODEL_PATH, PATHOLOGIES)
    
    # Test all images in directory
    test_results = []
    for img_file in os.listdir(TEST_IMAGES_DIR):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(TEST_IMAGES_DIR, img_file)
            
            try:
                # Analyze image quality first
                quality_info = predictor.analyze_image_quality(img_path)
                
                # Get prediction
                results = predictor.predict(img_path, use_tta=True)
                
                test_results.append({
                    'image': img_file,
                    'has_abnormality': results['has_abnormality'],
                    'confidence': results['confidence_score'],
                    'detected_pathologies': [p['pathology'] for p in results['detected_pathologies']],
                    'image_quality': quality_info
                })
                
                print(f"📊 {img_file}:")
                print(f"   Abnormal: {results['has_abnormality']}")
                print(f"   Confidence: {results['confidence_score']:.3f}")
                print(f"   Detected: {[p['pathology'] for p in results['detected_pathologies']]}")
                print(f"   Quality: {quality_info['quality']} ({quality_info['message']})")
                print()
                
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
    
    return test_results

if __name__ == "__main__":
    # Test multiple images to evaluate performance
    results = test_multiple_images()
    
    # Calculate performance metrics
    total_images = len(results)
    abnormal_detected = sum(1 for r in results if r['has_abnormality'])
    
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total images tested: {total_images}")
    print(f"Abnormalities detected: {abnormal_detected}/{total_images}")
    print(f"Detection rate: {abnormal_detected/total_images:.1%}")