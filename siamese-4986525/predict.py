import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from dataset import SiameseDataset
from model import SiameseNetwork
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
embedding_dim = 128

# Data transforms (MUST match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load trained model
model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
model.load_state_dict(torch.load("/workspace/Siamese/siamese_model_balanced.pth"))
model.eval()

print("Model loaded successfully!")

# Step 1: Load training data and split into train/validation
print("\n=== Loading Training Data ===")
train_labels_df = pd.read_csv(config.train_labels)

# Create binary labels
train_labels_df['binary_label'] = train_labels_df['diagnosis'].apply(lambda x: 1 if x == 'melanoma' else 0)

# Get image paths
all_files = os.listdir(config.train_image)
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_names = [f for f in all_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
image_dict = {os.path.splitext(name)[0]: name for name in image_names}

# Match images with labels
train_image_paths = []
train_image_labels = []

for _, row in train_labels_df.iterrows():
    image_id = str(row["image_name"]).strip()
    if image_id in image_dict:
        full_path = os.path.join(config.train_image, image_dict[image_id])
        train_image_paths.append(full_path)
        train_image_labels.append(row['binary_label'])

print(f"Total training images found: {len(train_image_paths)}")

# Split into reference set (80%) and validation set (20%)
ref_paths, val_paths, ref_labels, val_labels = train_test_split(
    train_image_paths, 
    train_image_labels, 
    test_size=0.2, 
    random_state=42,
    stratify=train_image_labels
)

print(f"Reference set size: {len(ref_paths)}")
print(f"Validation set size: {len(val_paths)}")

# Step 2: Create reference embeddings from reference set
print("\n=== Creating Reference Embeddings ===")

# Separate reference images by label
melanoma_ref_paths = [ref_paths[i] for i in range(len(ref_paths)) if ref_labels[i] == 1]
normal_ref_paths = [ref_paths[i] for i in range(len(ref_paths)) if ref_labels[i] == 0]

print(f"Melanoma reference images: {len(melanoma_ref_paths)}")
print(f"Normal reference images: {len(normal_ref_paths)}")

# Compute reference embeddings
def compute_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.forward_once(img)
        return embedding.cpu()
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

print("Computing melanoma reference embeddings...")
melanoma_embeddings = []
for path in melanoma_ref_paths:
    emb = compute_embedding(path)
    if emb is not None:
        melanoma_embeddings.append(emb)

melanoma_embeddings = torch.cat(melanoma_embeddings, dim=0) if melanoma_embeddings else None

print("Computing normal reference embeddings...")
normal_embeddings = []
for path in normal_ref_paths:
    emb = compute_embedding(path)
    if emb is not None:
        normal_embeddings.append(emb)

normal_embeddings = torch.cat(normal_embeddings, dim=0) if normal_embeddings else None

print(f"Melanoma embeddings shape: {melanoma_embeddings.shape}")
print(f"Normal embeddings shape: {normal_embeddings.shape}")

# Step 3: Predict on validation set and collect distances for ROC curve
print("\n=== Running Validation Predictions ===")
predictions = []
prediction_scores = []  # Distance scores for ROC curve
correct = 0
accuracy_over_time = []
sample_indices = []
all_embeddings = []  # For t-SNE visualization
all_labels = []  # For t-SNE visualization

melanoma_embeddings = melanoma_embeddings.to(device)
normal_embeddings = normal_embeddings.to(device)

for idx, (val_path, true_label) in enumerate(zip(val_paths, val_labels)):
    # Compute validation image embedding
    val_embedding = compute_embedding(val_path)
    
    if val_embedding is None:
        continue
    
    val_embedding = val_embedding.to(device)
    
    # Store embedding for t-SNE
    all_embeddings.append(val_embedding.cpu().numpy())
    all_labels.append(true_label)
    
    # Compare to melanoma references
    melanoma_distances = F.pairwise_distance(
        val_embedding.expand(melanoma_embeddings.size(0), -1),
        melanoma_embeddings
    )
    avg_melanoma_dist = torch.mean(melanoma_distances).item()
    
    # Compare to normal references
    normal_distances = F.pairwise_distance(
        val_embedding.expand(normal_embeddings.size(0), -1),
        normal_embeddings
    )
    avg_normal_dist = torch.mean(normal_distances).item()
    
    # Decision score: positive if closer to melanoma, negative if closer to normal
    # Higher score = more likely melanoma
    decision_score = avg_normal_dist - avg_melanoma_dist
    prediction_scores.append(decision_score)
    
    # Classify based on which reference set is closer
    prediction = 1 if avg_melanoma_dist < avg_normal_dist else 0
    predictions.append(prediction)
    
    # Check if correct
    if prediction == true_label:
        correct += 1
    
    # Track accuracy over time
    current_accuracy = correct / (idx + 1)
    accuracy_over_time.append(current_accuracy)
    sample_indices.append(idx + 1)
    
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(val_paths)} validation images, Current Accuracy: {current_accuracy:.4f}")

# Step 4: Calculate final accuracy and metrics
final_accuracy = correct / len(predictions)

print(f"\n{'='*50}")
print(f"FINAL VALIDATION ACCURACY: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"{'='*50}")
print(f"Correct predictions: {correct}/{len(predictions)}")

# Calculate confusion matrix metrics
true_positives = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and val_labels[i] == 1])
false_positives = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and val_labels[i] == 0])
true_negatives = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and val_labels[i] == 0])
false_negatives = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and val_labels[i] == 1])

print(f"\n=== Confusion Matrix ===")
print(f"True Positives (Melanoma correctly identified): {true_positives}")
print(f"False Positives (Normal classified as Melanoma): {false_positives}")
print(f"True Negatives (Normal correctly identified): {true_negatives}")
print(f"False Negatives (Melanoma classified as Normal): {false_negatives}")

if (true_positives + false_positives) > 0:
    precision = true_positives / (true_positives + false_positives)
    print(f"\nPrecision (Melanoma): {precision:.4f}")
else:
    precision = 0

if (true_positives + false_negatives) > 0:
    recall = true_positives / (true_positives + false_negatives)
    print(f"Recall (Sensitivity): {recall:.4f}")
else:
    recall = 0

if (true_negatives + false_positives) > 0:
    specificity = true_negatives / (true_negatives + false_positives)
    print(f"Specificity: {specificity:.4f}")

if precision > 0 and recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"F1 Score: {f1_score:.4f}")

melanoma_predicted = sum(predictions)
melanoma_actual = sum(val_labels)
print(f"\nMelanoma predicted: {melanoma_predicted}")
print(f"Melanoma actual: {melanoma_actual}")
print(f"Normal predicted: {len(predictions) - melanoma_predicted}")
print(f"Normal actual: {len(val_labels) - melanoma_actual}")

# Step 5: Generate all visualizations
print("\n=== Generating Visualizations ===")

# Create a 2x2 subplot figure
fig = plt.figure(figsize=(20, 16))

# Plot 1: Confusion Matrix Heatmap
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(val_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Melanoma'], 
            yticklabels=['Normal', 'Melanoma'],
            cbar_kws={'label': 'Count'})
ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 2: ROC Curve
ax2 = plt.subplot(2, 2, 2)
fpr, tpr, thresholds = roc_curve(val_labels, prediction_scores)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='#A23B72', lw=2, linestyle='--', label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy over time
ax3 = plt.subplot(2, 2, 3)
ax3.plot(sample_indices, accuracy_over_time, linewidth=2.5, color='#2E86AB', label='Validation Accuracy')
ax3.axhline(y=final_accuracy, color='#A23B72', linestyle='--', linewidth=2.5, label=f'Final Accuracy: {final_accuracy:.4f}')
ax3.set_xlabel('Number of Validation Samples', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_ylim([0, 1.0])

# Plot 4: t-SNE Visualization of Embeddings
print("Computing t-SNE...")
ax4 = plt.subplot(2, 2, 4)
all_embeddings_array = np.vstack(all_embeddings).squeeze()
all_labels_array = np.array(all_labels)

# Sample subset for faster t-SNE if dataset is large
max_samples = 1000
if len(all_embeddings_array) > max_samples:
    indices = np.random.choice(len(all_embeddings_array), max_samples, replace=False)
    all_embeddings_sample = all_embeddings_array[indices]
    all_labels_sample = all_labels_array[indices]
else:
    all_embeddings_sample = all_embeddings_array
    all_labels_sample = all_labels_array

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings_sample)

# Plot t-SNE
scatter = ax4.scatter(embeddings_2d[all_labels_sample == 0, 0], 
                      embeddings_2d[all_labels_sample == 0, 1],
                      c='#2E86AB', label='Normal', alpha=0.6, s=50)
scatter = ax4.scatter(embeddings_2d[all_labels_sample == 1, 0], 
                      embeddings_2d[all_labels_sample == 1, 1],
                      c='#A23B72', label='Melanoma', alpha=0.6, s=50)
ax4.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
ax4.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
ax4.set_title('t-SNE Visualization of Embeddings', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_validation_metrics.png', dpi=300, bbox_inches='tight')
print("Comprehensive validation plot saved as 'comprehensive_validation_metrics.png'")
plt.show()

# Save results to CSV
results_df = pd.DataFrame({
    'image_path': val_paths,
    'true_label': val_labels,
    'prediction': predictions,
    'prediction_score': prediction_scores,
    'correct': [predictions[i] == val_labels[i] for i in range(len(predictions))]
})

results_df.to_csv('validation_results.csv', index=False)
print("\nValidation results saved to 'validation_results.csv'")

print("\n=== Validation Complete ===")
