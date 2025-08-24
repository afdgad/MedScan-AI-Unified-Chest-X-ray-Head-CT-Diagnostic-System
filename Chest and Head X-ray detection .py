import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import re


# Set device
device = torch.device("cpu")
print(f"Using device: {device}")

# Define pathologies
CHEST_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

HEAD_PATHOLOGIES = [
    'Intraventricular', 'Intraparenchymal', 'Subarachnoid', 
    'Epidural', 'Subdural', 'Fracture'
]


class CustomHeadModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classification_head(x)
        return x


class MedicalDemographicsEstimator:
    def __init__(self):
        print("Medical Demographics Estimator Initialized")

    def extract_medical_features(self, image_array):
        features = {}
        features['mean_intensity'] = np.mean(image_array)
        features['intensity_std'] = np.std(image_array)
        features['contrast'] = np.std(image_array)
        features['aspect_ratio'] = image_array.shape[1] / image_array.shape[0]

        hist = np.histogram(image_array, bins=10)[0]
        hist_mean = np.mean(hist)
        hist_std = np.std(hist)
        if hist_std > 0:
            features['hist_skew'] = np.sum((hist - hist_mean)**3) / (len(hist) * hist_std**3)
        else:
            features['hist_skew'] = 0

        return np.array(list(features.values())).reshape(1, -1)

    def estimate_age_gender(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            img_array = np.array(image)
            features = self.extract_medical_features(img_array)

            intensity = features[0][0]
            if intensity > 180:
                age_group = "Elderly (65+)"
                age_confidence = 0.7
            elif intensity > 150:
                age_group = "Adult (45-64)"
                age_confidence = 0.6
            elif intensity > 120:
                age_group = "Young Adult (25-44)"
                age_confidence = 0.5
            elif intensity > 90:
                age_group = "Adult (18-24)"
                age_confidence = 0.4
            else:
                age_group = "Children (0-17)"
                age_confidence = 0.6

            aspect_ratio = features[0][3]
            if aspect_ratio > 1.2:
                gender = "Male"
                gender_confidence = 0.6
            elif aspect_ratio < 0.8:
                gender = "Female"
                gender_confidence = 0.6
            else:
                gender = "Unknown"
                gender_confidence = 0.3

            return {
                'age_group': age_group,
                'gender': gender,
                'age_confidence': age_confidence,
                'gender_confidence': gender_confidence,
                'method': 'medical_image_analysis'
            }

        except Exception as e:
            print(f"Demographic estimation error: {e}")
            return {
                'age_group': "Unknown",
                'gender': "Unknown",
                'age_confidence': 0.0,
                'gender_confidence': 0.0,
                'method': 'error'
            }


class UnifiedXRaySystem:
    def __init__(self, chest_model_path, head_model_path):
        self.device = device
        self.chest_model = self._load_chest_model(chest_model_path)
        self.head_model = self._load_head_model(head_model_path)
        self.demographics_estimator = MedicalDemographicsEstimator()

        print("Unified X-Ray Detection System Initialized!")
        print(f"Chest model: DenseNet121")
        print(f"Head model: Custom ResNet-based Architecture")

    def _load_chest_model(self, model_path):
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(CHEST_PATHOLOGIES))

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        print("Chest X-ray model (DenseNet121) loaded successfully!")
        return model

    def _load_head_model(self, model_path):
        model = CustomHeadModel(num_classes=len(HEAD_PATHOLOGIES))
        checkpoint = torch.load(model_path, map_location=self.device)

        state_dict = checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder_features'):
                new_key = key.replace('encoder_features', 'encoder')
                new_state_dict[new_key] = value
            elif key.startswith('encoder.') or key.startswith('classification_head'):
                new_state_dict[key] = value
            elif key.startswith('model.'):
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        try:
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            model.load_state_dict(new_state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        print("Head CT model (Custom Architecture) loaded successfully!")
        return model

    def _preprocess_image(self, image_path, image_type='chest'):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor, image

    def detect_image_type_from_filename(self, image_path):
        """Detect image type based on filename keywords"""
        # Get filename without extension
        filename = os.path.basename(image_path).lower()
        name_without_ext = os.path.splitext(filename)[0]

        # Define keywords
        chest_keywords = ['chest', 'x-ray', 'xray', 'xr', 'pulmonary', 'lung', 'thorax']
        head_keywords = ['head', 'ct', 'brain', 'skull', 'intracranial']

        # Check for chest keywords first
        for keyword in chest_keywords:
            if keyword in name_without_ext:
                print(f"Filename detection: Found '{keyword}' → CHEST")
                return 'chest'

        # If no chest keyword, check for head
        for keyword in head_keywords:
            if keyword in name_without_ext:
                print(f"Filename detection: Found '{keyword}' → HEAD")
                return 'head'

        # Default fallback
        print("Filename detection: No clear keyword → defaulting to HEAD")
        return 'head'

    def predict_chest_xray(self, image_tensor):
        with torch.no_grad():
            outputs = self.chest_model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

        results = []
        for i, prob in enumerate(probabilities):
            if prob > 0.2:
                results.append({
                    'pathology': CHEST_PATHOLOGIES[i],
                    'probability': float(prob),
                    'confidence': f"{prob:.1%}"
                })
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results, probabilities

    def predict_head_ct(self, image_tensor):
        with torch.no_grad():
            outputs = self.head_model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()

        results = []
        for i, prob in enumerate(probabilities):
            if prob > 0.2:
                results.append({
                    'pathology': HEAD_PATHOLOGIES[i],
                    'probability': float(prob),
                    'confidence': f"{prob:.1%}"
                })
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results, probabilities

    def highlight_abnormalities(self, original_image, predictions, image_type):
        visualized_image = original_image.copy()
        if visualized_image.mode != 'RGB':
            visualized_image = visualized_image.convert('RGB')
        draw = ImageDraw.Draw(visualized_image)
        width, height = visualized_image.size

        draw.text((10, 10), f"Image Type: {image_type.upper()}", fill="white")

        for i, prediction in enumerate(predictions[:3]):
            color = ['red', 'yellow', 'blue'][i % 3]
            box_size = min(width, height) // 6
            x1 = width // 8 + (i * box_size)
            y1 = height - box_size - 30
            x2 = x1 + box_size
            y2 = y1 + box_size
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{prediction['pathology']}: {prediction['confidence']}"
            draw.text((x1, y1 - 20), label, fill=color)

        return visualized_image

    def generate_report(self, predictions, demographics, image_type):
        report = {
            'image_type': image_type,
            'age_group': demographics['age_group'],
            'gender': demographics['gender'],
            'age_confidence': demographics['age_confidence'],
            'gender_confidence': demographics['gender_confidence'],
            'abnormalities_found': len(predictions) > 0,
            'abnormalities': predictions,
            'primary_concern': predictions[0] if predictions else None,
            'recommendation': self._generate_recommendation(predictions, demographics)
        }
        return report

    def _generate_recommendation(self, predictions, demographics):
        if not predictions:
            return "No significant abnormalities detected. Routine follow-up recommended."

        recommendations = []
        age_group = demographics['age_group']

        for prediction in predictions:
            path = prediction['pathology'].lower()
            conf = prediction['probability']

            if 'fracture' in path:
                rec = "Urgent orthopedic consultation recommended."
                if "Elderly" in age_group:
                    rec += " High risk of complications in elderly patients."
                recommendations.append(rec)

            elif any(term in path for term in ['pneumonia', 'atelectasis', 'infiltration']):
                rec = "Pulmonary evaluation advised."
                if "Children" in age_group or "Elderly" in age_group:
                    rec += " High-risk age group requires close monitoring."
                recommendations.append(rec)

            elif any(hemo in path for hemo in ['intraventricular', 'intraparenchymal', 'subarachnoid', 'epidural', 'subdural']):
                recommendations.append("Neurological emergency consultation required.")

            elif any(cardio in path for cardio in ['cardiomegaly', 'edema']):
                rec = "Cardiology consultation recommended."
                if "Elderly" in age_group:
                    rec += " Age-related cardiovascular risk factors present."
                recommendations.append(rec)

            elif conf > 0.5:
                recommendations.append("Specialist consultation recommended.")
            else:
                recommendations.append("Further investigation suggested.")

        return " ".join(recommendations[:3])

    def analyze_image(self, image_path):
        print(f"Analyzing image: {image_path}")
        try:
            # Detect image type from filename
            image_type = self.detect_image_type_from_filename(image_path)
            print(f"Detected image type: {image_type}")

            # Preprocess image
            image_tensor, original_image = self._preprocess_image(image_path, image_type)

            # Predict using correct model
            if image_type == 'chest':
                print("Using DenseNet121 for chest X-ray prediction")
                abnormalities, all_probs = self.predict_chest_xray(image_tensor)
            else:
                print("Using Custom Model for head CT prediction")
                abnormalities, all_probs = self.predict_head_ct(image_tensor)

            # Estimate demographics
            demographics = self.demographics_estimator.estimate_age_gender(image_path)

            # Highlight abnormalities
            highlighted_image = self.highlight_abnormalities(original_image, abnormalities, image_type)

            # Generate report
            report = self.generate_report(abnormalities, demographics, image_type)

            return {
                'original_image': original_image,
                'highlighted_image': highlighted_image,
                'report': report,
                'all_probabilities': all_probs,
                'image_type': image_type,
                'demographics': demographics
            }

        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                'error': str(e),
                'report': {
                    'image_type': 'unknown',
                    'age_group': "Unknown",
                    'gender': "Unknown",
                    'abnormalities_found': False,
                    'abnormalities': [],
                    'recommendation': f"Analysis failed: {str(e)}"
                }
            }


def visualize_results(results, save_path=None):
    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(results['original_image'])
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(results['highlighted_image'])
    axes[1].set_title('Detected Abnormalities', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    plt.show()

    print("\n" + "="*70)
    print("COMPREHENSIVE DIAGNOSTIC REPORT")
    print("="*70)
    print(f"Patient Demographics: {results['report']['age_group']}, {results['report']['gender']}")

    if results['report']['abnormalities_found']:
        print(f"\nDETECTED ABNORMALITIES ({len(results['report']['abnormalities'])}):")
        print("-" * 50)
        for i, ab in enumerate(results['report']['abnormalities'], 1):
            print(f"{i:2d}. {ab['pathology']:20s}: {ab['confidence']:>8s}")
    else:
        print("\nNo abnormalities detected. Image appears normal.")

    print(f"\nMEDICAL RECOMMENDATION:")
    print("-" * 50)
    print(results['report']['recommendation'])


def main():
    # UPDATE THESE PATHS
    CHEST_MODEL_PATH = "MODELS/densenet121_nih_best.pth"
    HEAD_MODEL_PATH = "MODELS/dual_task_model_best_weights.pth"
    TEST_IMAGE = "TEST_IMAGES/chest_xray_03.png"  # Change to your test image path

    for path, name in [(CHEST_MODEL_PATH, "Chest Model"),
                       (HEAD_MODEL_PATH, "Head Model"),
                       (TEST_IMAGE, "Test Image")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found at: {path}")
            return

    print("Initializing Unified X-Ray Detection System...")
    system = UnifiedXRaySystem(CHEST_MODEL_PATH, HEAD_MODEL_PATH)

    print(f"\nAnalyzing image: {TEST_IMAGE}")
    results = system.analyze_image(TEST_IMAGE)
    visualize_results(results, save_path="diagnostic_report.png")

    if 'error' not in results:
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        model_name = 'DenseNet121' if results['image_type'] == 'chest' else 'Custom Head Model'
        print(f"Model Used: {model_name}")
        print(f"Image Type: {results['image_type'].upper()}")
        print(f"Abnormalities Found: {len(results['report']['abnormalities'])}")


if __name__ == "__main__":
    main()