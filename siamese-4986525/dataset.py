"""
Dataset.py

This module handles the loading, pairing, and balancing of the ISIC 2020 dataset
for a Siamese network-based skin lesion classification task.

Author: Wu Chun Yueh 
Date: 31th October 2025
"""
import random
import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image

class SiameseDataset(Dataset):
    
    """
    Custom Dataset class for creating Siamese network image pairs from the ISIC 2020 dataset
    Handles loading of images, creation of positive and negative pairs, and balancing of classes
    through oversampling of the minority class (melanoma)
    
    Args:
        image_dir (str): Directory containing the dataset images.
        labels_csv (str): Path to the CSV file containing image labels and diagnoses
        transform (callable, optional): Optional transform to be applied on samples. Typically normalization and resizing
        augment_transform (callable, optional): Optional augmentation transform specifcly for melnoma samples
        has_labels (bool): Whether the dataset includes labels (True for training, False for test)
        balance_classes (bool): Whether to balance the dataset by oversampling the minority class (melanoma)
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] 

    def __init__(self, image_dir, labels_csv, transform=None, augment_transform=None, has_labels=True, balance_classes=True):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.augment_transform = augment_transform  # Augmentation for melanoma oversampling
        self.has_labels = has_labels
        self.balance_classes = balance_classes

        if self.has_labels:
            self.create_labels()
            self.pairs = self.pairs_image_label()
            
            # Balance dataset if requested
            if self.balance_classes:
                self.pairs = self.balance_dataset()
        else:
            self.pairs = self.load_test_images()

    def is_image_file(self, filename):

        """
        Check if a file has a valid image extension Ex: img1.jpg, img2.png
            
        Returns:
            bool: True if the file has a valid image extension, False otherwise.
        """
        return any(filename.lower().endswith(ext) for ext in self.valid_extensions)

    def create_labels(self):
        """
        Convert diagnosis labels from string format to binary labels.
        
        Labels:
            0 = Normal (benign)
            1 = Melanoma (malignant)
            
        Adds a new column 'binary_label' to the labels dataframe.
        """
        labels = []
        for i in range(len(self.labels_df)):
            if self.labels_df.loc[i, "diagnosis"] == "melanoma":
                labels.append(1)
            else:
                labels.append(0)
        self.labels_df['binary_label'] = labels

    def pairs_image_label(self):
        """
        Read folder and match each images file with its corresponding lables from the CSV. Create pairs for training
        
        Returns:
            list: List of tuples (image_path, label) for all valid images found in the directory.
        """

        all_files = os.listdir(self.image_dir)
        image_names = [f for f in all_files if os.path.isfile(os.path.join(self.image_dir, f)) and self.is_image_file(f)]
        image_dict = {os.path.splitext(name)[0]: name for name in image_names}

        pairs = []
        for _, row in self.labels_df.iterrows():
            image_id = str(row["image_name"]).strip()
            label = row["binary_label"]

            if image_id in image_dict:
                full_path = os.path.join(self.image_dir, image_dict[image_id])
                pairs.append((full_path, label))

        print(f"Loaded {len(pairs)} image pairs from {self.image_dir}")
        return pairs

    def balance_dataset(self):
        """
        Balance the dataset by oversampling the minority class (melanoma)

        Each melanoma image will be duplicated with augmentations until matches the size of the normal class

         Returns:
            list: Balanced list of tuples (image_path, label, is_augmented) where:
                  - image_path: Full path to the image file
                  - label: Binary label (0 for normal, 1 for melanoma)
                  - is_augmented: Boolean indicating if image should be augmented during loading
        """
        melanoma_pairs = [(path, label) for path, label in self.pairs if label == 1]
        normal_pairs = [(path, label) for path, label in self.pairs if label == 0]
        
        num_melanoma = len(melanoma_pairs)
        num_normal = len(normal_pairs)
        
        print(f"\nOriginal dataset:")
        print(f"  Melanoma: {num_melanoma}")
        print(f"  Normal: {num_normal}")
        print(f"  Ratio: 1:{num_normal/num_melanoma:.1f}")
        
        # Calculate how many times to duplicate melanoma images
        if num_melanoma < num_normal:
            oversample_factor = num_normal // num_melanoma
            remainder = num_normal % num_melanoma
            
            # Create augmented copies
            balanced_melanoma = []
            for i, (path, label) in enumerate(melanoma_pairs):
                # Add original
                balanced_melanoma.append((path, label, False))  # (path, label, is_augmented)
                
                # Add augmented copies
                for _ in range(oversample_factor - 1):
                    balanced_melanoma.append((path, label, True))
            
            # Add remainder augmented samples
            for i in range(remainder):
                path, label = melanoma_pairs[i % num_melanoma]
                balanced_melanoma.append((path, label, True))
            
            # Combine with normal images (mark as not augmented)
            balanced_pairs = balanced_melanoma + [(path, label, False) for path, label in normal_pairs]
            
            print(f"\nBalanced dataset:")
            print(f"  Melanoma: {len(balanced_melanoma)} (original: {num_melanoma}, augmented: {len(balanced_melanoma) - num_melanoma})")
            print(f"  Normal: {num_normal}")
            print(f"  Total: {len(balanced_pairs)}")
            
            return balanced_pairs
        else:
            # If already balanced or melanoma is majority (unlikely), return as is
            return [(path, label, False) for path, label in self.pairs]

    def load_test_images(self):
        """
        Load test images from directory without labels and label -1

        Replace unlabeled images with predicted labels 
        Returns:
            list: List of tuples (image_path, label, is_augmented) with label = -1 for test images.
        """
        all_files = os.listdir(self.image_dir)
        image_names = [f for f in all_files if os.path.isfile(os.path.join(self.image_dir, f)) and self.is_image_file(f)]
        pairs = [(os.path.join(self.image_dir, img), -1, False) for img in image_names]
        print(f"Loaded {len(pairs)} images from {self.image_dir} without labels")
        return pairs

    def __len__(self):
      """Return the total number of samples in the dataset."""
        return len(self.pairs)

    def __getitem__(self, index):
        """
        Fetch a pair of images for Siamese network training or a single image for testing.
        
        For training (has_labels=True):
            Returns either a positive pair (same class) or negative pair (different classes) based on random selection.
            Transforms and augmentate the images if required
            
        For testing (has_labels=False):
            Returns a single image with dummy label -1
        
        Args:
            index (int): Index of the first image in the pair.
            
        Returns:
            For training mode:
                tuple: (image_1, image_2, pair_label) where:
                    - image_1, image_2: Transformed PIL Images or tensors
                    - pair_label: 1 for positive pair (same class), 0 for negative pair (different classes)
                    
            For test mode:
                tuple: (image_1, -1) where image_1 is the transformed image
        """
        img_1_path, label_1, is_augmented_1 = self.pairs[index]
        img_1 = Image.open(img_1_path).convert("RGB")
        
        # Add augmentation if this is an augmented melanoma sample
        if is_augmented_1 and self.augment_transform is not None and label_1 == 1:
            img_1 = self.augment_transform(img_1)
        elif self.transform is not None:
            img_1 = self.transform(img_1)

        if self.has_labels:
            # Create a pair with label
            same_class = random.randint(0, 1)
            
            # Find matching pair
            while True:
                idx_2 = random.randint(0, len(self.pairs) - 1)
                img_2_path, label_2, is_augmented_2 = self.pairs[idx_2]
                
                if (same_class and label_1 == label_2) or (not same_class and label_1 != label_2):
                    pairs_label = int(same_class)
                    break

            img_2 = Image.open(img_2_path).convert("RGB")
            
            # Apply augmentation if this is an augmented melanoma sample
            if is_augmented_2 and self.augment_transform is not None and label_2 == 1:
                img_2 = self.augment_transform(img_2)
            elif self.transform is not None:
                img_2 = self.transform(img_2)

            return img_1, img_2, pairs_label
        else:
            # Test mode: return single image and dummy label
            return img_1, -1
