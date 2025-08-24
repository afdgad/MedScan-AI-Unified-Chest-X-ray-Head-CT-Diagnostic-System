import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

# Paths
base_path = r"archive (1)/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"
patients_path = os.path.join(base_path, "Patients_CT")
demographics_path = os.path.join(base_path, "patient_demographics.csv")
diagnosis_path = os.path.join(base_path, "hemorrhage_diagnosis.csv")

# Load metadata
def explore_dataset():
    print("=== DATASET EXPLORATION ===")
    
    # 1. Load demographic data
    demographics = pd.read_csv(demographics_path)
    print("Patient Demographics:")
    print(demographics.head())
    print(f"Total patients: {len(demographics)}")
    print(f"Age range: {demographics['Age(years)'].min()} - {demographics['Age(years)'].max()}")
    print(f"Gender distribution:\n{demographics['Gender'].value_counts()}")
    
    # 2. Load diagnosis data
    diagnosis = pd.read_csv(diagnosis_path)
    print("\nHemorrhage Diagnosis:")
    print(diagnosis.head())
    print(f"Total slices with labels: {len(diagnosis)}")
    
    # 3. Check hemorrhage types
    hemorrhage_types = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
    for ht in hemorrhage_types:
        count = diagnosis[ht].sum()
        print(f"{ht}: {count} slices")
    
    print(f"Fracture cases: {diagnosis["Fracture_Yes_No"].sum()}")
    
    # 4. Check actual image files
    patient_folders = [f for f in os.listdir(patients_path) 
                      if os.path.isdir(os.path.join(patients_path, f))]
    
    print(f"\nFound {len(patient_folders)} patient folders")
    
    # Check one patient
    sample_patient = patient_folders[0]
    sample_brain_path = os.path.join(patients_path, sample_patient, "brain")
    
    if os.path.exists(sample_brain_path):
        brain_images = [f for f in os.listdir(sample_brain_path) if f.endswith('.jpg')]
        ct_images = [f for f in brain_images if 'HGE_Seg' not in f]
        seg_masks = [f for f in brain_images if 'HGE_Seg' in f]
        
        print(f"Sample patient {sample_patient}:")
        print(f"  CT images: {len(ct_images)}")
        print(f"  Segmentation masks: {len(seg_masks)}")
        
        # Display sample
        if ct_images and seg_masks:
            ct_sample = os.path.join(sample_brain_path, ct_images[0])
            seg_sample = os.path.join(sample_brain_path, seg_masks[0])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # CT image
            ct_img = Image.open(ct_sample)
            ax1.imshow(ct_img, cmap='gray')
            ax1.set_title('CT Brain Window')
            ax1.axis('off')
            
            # Segmentation mask
            seg_img = Image.open(seg_sample)
            ax2.imshow(seg_img, cmap='hot')
            ax2.set_title('Hemorrhage Segmentation')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()

# Run exploration
explore_dataset()