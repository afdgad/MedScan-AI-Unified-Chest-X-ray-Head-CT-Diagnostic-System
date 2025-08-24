import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
base_path = r"archive (1)\computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"
patients_path = os.path.join(base_path, "Patients_CT")
demographics_path = os.path.join(base_path, "patient_demographics.csv")
diagnosis_path = os.path.join(base_path, "hemmorhage_diagnosis.csv")

# Define hemorrhage types
HEMORRHAGE_TYPES = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
NUM_CLASSES = len(HEMORRHAGE_TYPES) + 1  # +1 for fracture

# Combined Model Architecture
class DualTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(DualTaskModel, self).__init__()
        
        # Encoder (shared backbone) - Using ResNet50
        self.encoder = models.resnet50(pretrained=True)
        
        # Remove original fully connected layer
        self.encoder_features = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Segmentation Decoder (U-Net like)
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # Output: 1 channel for segmentation
            nn.Sigmoid()
        )
        
        # Classification Head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Shared encoder
        features = self.encoder_features(x)
        
        # Segmentation branch
        seg_output = self.seg_decoder(features)
        
        # Classification branch
        cls_output = self.classification_head(features)
        
        return seg_output, cls_output

# Dataset Class
class HeadCTDualTaskDataset(Dataset):
    def __init__(self, patients_path, diagnosis_df, transform=None, target_size=(256, 256)):
        self.patients_path = patients_path
        self.diagnosis_df = diagnosis_df
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        
        self._prepare_samples()
        print(f"Loaded {len(self.samples)} samples for dual-task training")
    
    def _prepare_samples(self):
        """Prepare samples with both classification and segmentation labels"""
        for idx, row in self.diagnosis_df.iterrows():
            patient_id = f"Patient_{int(row['PatientNumber']):03d}"
            slice_num = int(row['SliceNumber'])
            
            # CT image path
            ct_path = os.path.join(self.patients_path, patient_id, "brain", f"{slice_num}.jpg")
            
            # Segmentation mask path
            seg_path = os.path.join(self.patients_path, patient_id, "brain", f"{slice_num}_HGE_Seg.jpg")
            
            if os.path.exists(ct_path):
                # Classification labels
                hemorrhage_labels = row[HEMORRHAGE_TYPES].values.astype(np.float32)
                fracture_label = np.array([row['Fracture']], dtype=np.float32)
                classification_label = np.concatenate([hemorrhage_labels, fracture_label])
                
                self.samples.append({
                    'ct_path': ct_path,
                    'seg_path': seg_path,
                    'classification_label': classification_label,
                    'patient_id': patient_id,
                    'slice_num': slice_num
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load CT image
            ct_image = Image.open(sample['ct_path']).convert('L')  # Grayscale
            ct_image = ct_image.resize(self.target_size)
            
            # Convert to 3-channel for pretrained model
            ct_image = ct_image.convert('RGB')
            
            # Load segmentation mask (create empty if not exists)
            if os.path.exists(sample['seg_path']):
                seg_mask = Image.open(sample['seg_path']).convert('L')
                seg_mask = seg_mask.resize(self.target_size, Image.NEAREST)
                seg_mask = np.array(seg_mask) > 0  # Binary mask
            else:
                seg_mask = np.zeros(self.target_size, dtype=bool)
            
            # Apply transformations
            if self.transform:
                ct_image = self.transform(ct_image)
            
            return {
                'image': ct_image,
                'seg_mask': torch.tensor(seg_mask, dtype=torch.float32).unsqueeze(0),
                'cls_label': torch.tensor(sample['classification_label'], dtype=torch.float32),
                'patient_id': sample['patient_id'],
                'slice_num': sample['slice_num']
            }
            
        except Exception as e:
            print(f"Error loading sample {sample['ct_path']}: {e}")
            # Return dummy sample
            return {
                'image': torch.zeros(3, *self.target_size),
                'seg_mask': torch.zeros(1, *self.target_size),
                'cls_label': torch.zeros(NUM_CLASSES),
                'patient_id': 'dummy',
                'slice_num': -1
            }

# Loss Functions
class DualTaskLoss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super(DualTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss = nn.BCELoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, seg_pred, cls_pred, seg_target, cls_target):
        seg_loss = self.seg_loss(seg_pred, seg_target)
        cls_loss = self.cls_loss(cls_pred, cls_target)
        
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return total_loss, seg_loss, cls_loss

# Training Functions
def train_dual_task_model():
    """Main training function for dual-task learning"""
    
    # Load and prepare data
    diagnosis_df = pd.read_csv(diagnosis_path)
    
    # Create binary labels
    diagnosis_df['has_hemorrhage'] = diagnosis_df[HEMORRHAGE_TYPES].sum(axis=1) > 0
    
    # Split data
    train_df, val_df = train_test_split(
        diagnosis_df, test_size=0.2, random_state=42, stratify=diagnosis_df['has_hemorrhage']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HeadCTDualTaskDataset(patients_path, train_df, transform=train_transform)
    val_dataset = HeadCTDualTaskDataset(patients_path, val_df, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize model
    model = DualTaskModel(NUM_CLASSES).to(device)
    
    # Loss and optimizer
    criterion = DualTaskLoss(seg_weight=1.0, cls_weight=0.5)  # Adjust weights as needed
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 25
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            seg_masks = batch['seg_mask'].to(device)
            cls_labels = batch['cls_label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            seg_pred, cls_pred = model(images)
            
            # Compute loss
            loss, seg_loss, cls_loss = criterion(seg_pred, cls_pred, seg_masks, cls_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_cls_loss += cls_loss.item()
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                seg_masks = batch['seg_mask'].to(device)
                cls_labels = batch['cls_label'].to(device)
                
                seg_pred, cls_pred = model(images)
                loss, seg_loss, cls_loss = criterion(seg_pred, cls_pred, seg_masks, cls_labels)
                
                val_total_loss += loss.item()
                val_seg_loss += seg_loss.item()
                val_cls_loss += cls_loss.item()
        
        # Calculate averages
        train_total_loss /= len(train_loader)
        train_seg_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        
        val_total_loss /= len(val_loader)
        val_seg_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_total_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_total_loss:.4f} (Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f})')
        print(f'  Val Loss: {val_total_loss:.4f} (Seg: {val_seg_loss:.4f}, Cls: {val_cls_loss:.4f})')
        
        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'dual_task_model_best.pth')
            print('  ↳ Saved best model!')
    
    print('Training completed!')

# Visualization function
def visualize_predictions(model, dataloader, device, num_samples=5):
    """Visualize model predictions"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            seg_masks = batch['seg_mask']
            cls_labels = batch['cls_label']
            
            seg_pred, cls_pred = model(images)
            
            # Convert to numpy
            images = images.cpu().numpy()
            seg_masks = seg_masks.numpy()
            seg_pred = seg_pred.cpu().numpy()
            cls_pred = torch.sigmoid(cls_pred).cpu().numpy()
            cls_labels = cls_labels.numpy()
            
            for i in range(min(num_samples, len(images))):
                samples.append({
                    'image': images[i].transpose(1, 2, 0),
                    'true_mask': seg_masks[i][0],
                    'pred_mask': seg_pred[i][0] > 0.5,  # Threshold
                    'true_cls': cls_labels[i],
                    'pred_cls': cls_pred[i] > 0.5  # Threshold
                })
            
            if len(samples) >= num_samples:
                break
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, 3*num_samples))
    
    for i, sample in enumerate(samples):
        # Original image
        axes[i, 0].imshow(sample['image'], cmap='gray')
        axes[i, 0].set_title('CT Image')
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(sample['true_mask'], cmap='hot')
        axes[i, 1].set_title('True Hemorrhage')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(sample['pred_mask'], cmap='hot')
        axes[i, 2].set_title('Predicted Hemorrhage')
        axes[i, 2].axis('off')
        
        # Classification results
        axes[i, 3].text(0.1, 0.5, 
                       f"True: {sample['true_cls']}\nPred: {sample['pred_cls']}",
                       fontsize=10, transform=axes[i, 3].transAxes)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the model
    train_dual_task_model()
    
    # Load best model for visualization
    model = DualTaskModel(NUM_CLASSES).to(device)
    checkpoint = torch.load('dual_task_model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create validation loader for visualization
    diagnosis_df = pd.read_csv(diagnosis_path)
    val_df = diagnosis_df.sample(10, random_state=42) 
