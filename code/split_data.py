import os
from glob import glob
import sys
import shutil
SEED = 42
from sklearn.model_selection import train_test_split
# run: py split_train_val_test.py <path_folder> ex: py split_data.py
# split 80% train 10% test 10% val

def split_train_val_test(input_image_dir, input_labels_dir, output_dir):
    """YOLO format

    Args:
        input_image_dir (str): input image directory
        input_labels_dir (str): labels directory
        output_dir (str): output directory
    """
    output_train_dir = os.path.join(output_dir, "images/train")
    output_val_dir = os.path.join(output_dir, "images/val")
    output_test_dir = os.path.join(output_dir, "images/test")

    output_labels_train_dir = os.path.join(output_dir, "labels/train")
    output_labels_val_dir = os.path.join(output_dir, "labels/val")
    output_labels_test_dir = os.path.join(output_dir, "labels/test")

    if os.path.isdir(output_train_dir):
        shutil.rmtree(output_train_dir)

    if os.path.isdir(output_val_dir):
        shutil.rmtree(output_val_dir)
    
    if os.path.isdir(output_test_dir):
        shutil.rmtree(output_test_dir)

    if os.path.isdir(output_labels_train_dir):
        shutil.rmtree(output_labels_train_dir)

    if os.path.isdir(output_labels_val_dir):
        shutil.rmtree(output_labels_val_dir)
    
    if os.path.isdir(output_labels_test_dir):
        shutil.rmtree(output_labels_test_dir)
    os.makedirs(output_train_dir)
    os.makedirs(output_val_dir)
    os.makedirs(output_test_dir)
    os.makedirs(output_labels_train_dir)
    os.makedirs(output_labels_val_dir)
    os.makedirs(output_labels_test_dir)

    image_glob = sorted(glob(os.path.join(input_image_dir, "*.jpg")))
    label_glob = sorted(glob(os.path.join(input_labels_dir, "*.txt")))
    
    ## 80% train data, 20% (val, test) 
    train_img, temp_img, train_label, temp_label = train_test_split(
        image_glob, label_glob, train_size=0.8, shuffle=True, random_state=SEED
    )
    #split temp to test and val
    val_img, test_img, val_label, test_label = train_test_split(
        temp_img, temp_label, train_size=0.5, shuffle=True, random_state=SEED
    )
    for img, label in zip(train_img, train_label):
        shutil.copy(img, output_train_dir)
        shutil.copy(label, output_labels_train_dir)

    for img, label in zip(val_img, val_label):
        shutil.copy(img, output_val_dir)
        shutil.copy(label, output_labels_val_dir)

    for img, label in zip(test_img, test_label):
        shutil.copy(img, output_test_dir)
        shutil.copy(label, output_labels_test_dir)
if __name__ == "__main__":
    input_image_dir = "./images"
    input_labels_dir = "./labels"
    output_dir = "./split_data"

    split_train_val_test(input_image_dir, input_labels_dir, output_dir)

    

