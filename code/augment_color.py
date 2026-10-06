#! /usr/bin/env python3 
# -*- coding: utf-8 -*- 

# Standard library
import itertools
from pathlib import Path
import os
# External library
import albumentations as A
import cv2
import numpy as np
# Internal library

class YOLOAugmentationGenerator:
    
    def __init__(self, images_path, labels_path, output_path):
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path)
        self.output_path = Path(output_path)
        
        # Create output directories
        os.makedirs(self.output_path / 'images', exist_ok=True)
        os.makedirs(self.output_path / 'labels', exist_ok=True)
        
        
        self.color_augmentations = {
            'color_jitter': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        }
    
        self.flip_augmentations = {
            'horizontal_flip': A.HorizontalFlip(p=1),
            'vertical_flip': A.VerticalFlip(p=1),
        }
        
        self.rotation_augmentations = {
            # Rotate 90,180,270 degrees
            'rotate_90': A.Affine(rotate=90, p=1, mode=cv2.BORDER_CONSTANT, fit_output=True),
            'rotate_180': A.Affine(rotate=180, p=1, mode=cv2.BORDER_CONSTANT, fit_output=True),
            'rotate_270': A.Affine(rotate=270, p=1, mode=cv2.BORDER_CONSTANT, fit_output=True),
        }
        
    def read_yolo_label(self, label_path):
        boxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))
        return np.array(boxes), class_labels
    
    def save_yolo_label(self, boxes, class_labels, label_path):
        with open(label_path, 'w') as f:
            for box, class_id in zip(boxes, class_labels):
                f.write(f"{class_id} {' '.join(map(str, box))}\n")
                
    def apply_color_augmentation(self, image_path, label_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, class_labels = self.read_yolo_label(label_path)
        
        # Apply color augmentation
        transform = A.Compose(
            [self.color_augmentations['color_jitter']],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)
        
        # Generate output filename with augmentation information
        output_image_path = self.output_path / 'images' / f"{image_path.stem}_color.bmp"
        output_label_path = self.output_path / 'labels' / f"{image_path.stem}_color.txt"
        
        # Save augmented image and labels
        cv2.imwrite(str(output_image_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
        self.save_yolo_label(transformed['bboxes'], transformed['class_labels'], output_label_path)
        
    def apply_flip_augmentation(self, image_path, label_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, class_labels = self.read_yolo_label(label_path)
        
        # Apply flip augmentation
        for flip_name, flip_transform in self.flip_augmentations.items():
            transform = A.Compose(
                [flip_transform],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )
            transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)
            
            # Generate output filename with augmentation information
            output_image_path = self.output_path / 'images' / f"{image_path.stem}_{flip_name}.bmp"
            output_label_path = self.output_path / 'labels' / f"{image_path.stem}_{flip_name}.txt"
            
            # Save augmented image and labels
            cv2.imwrite(str(output_image_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            self.save_yolo_label(transformed['bboxes'], transformed['class_labels'], output_label_path)
            
    def apply_rotation_augmentation(self, image_path, label_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, class_labels = self.read_yolo_label(label_path)
        
        # Apply each rotation
        for rotation_name, rotation_transform in self.rotation_augmentations.items():
            transform = A.Compose(
                [rotation_transform],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )
            transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)
            
            # Generate output filename with augmentation information
            output_image_path = self.output_path / 'images' / f"{image_path.stem}_{rotation_name}.bmp"
            output_label_path = self.output_path / 'labels' / f"{image_path.stem}_{rotation_name}.txt"
            
            # Save augmented image and labels
            cv2.imwrite(str(output_image_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            self.save_yolo_label(transformed['bboxes'], transformed['class_labels'], output_label_path)
            
    def convert_png_to_jpg(self, image_path):
        image = cv2.imread(str(image_path))
        output_image_path = self.output_path / 'images' / f"{image_path.stem}.jpg"
        cv2.imwrite(str(output_image_path), image)
                
    def generate_all_augmentations(self):
        image_files = list(self.images_path.glob('*.jpg')) + list(self.images_path.glob('*.bmp'))

        for image_path in image_files:
            label_path = self.labels_path / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # apply augmentations
            # self.apply_color_augmentation(image_path, label_path)
            self.apply_flip_augmentation(image_path, label_path)
            # self.apply_rotation_augmentation(image_path, label_path)
            # self.convert_png_to_jpg(image_path)

# Usage example
images_path = r'D:\Hachix\note_task\alphavina\korea\bk_data\30122024\images'
labels_path = r'D:\Hachix\note_task\alphavina\korea\bk_data\30122024\labels'
output_path = r'D:\Hachix\note_task\alphavina\korea\bk_data\30122024\temp_flip'

augmentor = YOLOAugmentationGenerator(
    images_path=images_path,
    labels_path=labels_path,
    output_path=output_path
)
augmentor.generate_all_augmentations()
# from glob import glob
# for image_path in glob(os.path.join(images_path, "*.jpeg")):
#     image = cv2.imread(str(image_path))
#     cv2.imwrite(os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace(".jpeg", ".bmp")), image)