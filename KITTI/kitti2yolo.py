import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
from glob import glob
import os, json, sys

from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Visualization 1 image
def eda(img : str, labels : list = None) -> None:
    img = cv2.imread(img, cv2.IMREAD_ANYCOLOR)
    if labels is not None:
        for label in labels:
            cx, cy, w, h = label['bbox']
            min_x, min_y, max_x, max_y = \
                        int(cx-w//2), int(cy-h//2), int(cx+w//2), int(cy+h//2)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255,255,0), 2)
            cv2.putText(img, label['name'], org=(min_x,min_y), 
                            fontFace=cv2.FONT_ITALIC, fontScale=1, color=(0,255,0), thickness=2)
    cv2.imshow("img", img)
    cv2.waitKey(0)

# Visualization 20 image
def eda_subplot(label_list : list) -> None:
    plt.figure(figsize=(10,8)) # width, height
    for i,file in enumerate(label_list):
        plt.subplot(4,5,i+1)
        img_path = file.split('\\')[-1].split('.')[0]
        img = "./images/train/" + img_path + ".png"
        img = Image.open(img)
        img = np.array(img, np.uint8)
        plt.imshow(img)
    plt.show()

# yes/no visualization
visualization = False

# convert min_x, min_y, max_x, max_y => center_x, center_y, width, height
def xyxy2xywh_np(bbox, img_width, img_height):
    min_x, min_y, max_x, max_y = np.array(bbox, dtype=np.float32)

    center_x = round((max_x + min_x) / 2,2)
    center_y = round((max_y + min_y) / 2,2)
    bbox_width = round(max_x - min_x,2)
    bbox_height = round(max_y - min_y,2)

    yolo_x = center_x / img_width
    yolo_y = center_y / img_height
    yolo_width = bbox_width / img_width
    yolo_height = bbox_height / img_height

    bbox = (yolo_x, yolo_y, yolo_width, yolo_height)

    return bbox

# initial channels about grayscale
channels = 1

# convert kitti label to yolo label format
class convert2yolo():
    def __init__(self):
        self.label_dir = "./kitti_labels/val/"
        self.img_dir = "./images/"
        self.img_train_dir = self.img_dir + "val/"
        # self.img_valid_dir = self.img_dir + "valid/"
        self.output_dir = "./labels/val/"

        self.class_names = {
                        'Car' : 0, 
                        'Van' : 1, 
                        'Truck' : 2,
                        'Pedestrian' : 3, 
                        'Person_sitting' : 4, 
                        'Cyclist' : 5, 
                        'Tram' : 6,
                        'Misc' : 7,
                        'DontCare' : 8
                    }

        self.label_dir_list = glob(self.label_dir + "/*")
        os.makedirs(self.output_dir, exist_ok=True)

    # save the txt file for yolo label format to convert from kitti label format
    def save(self):
        for file in self.label_dir_list:
            # labels = []
            img_path = file.split('/')[-1].split('.')[0]
            img_name = self.img_train_dir + img_path + ".png"
            img = cv2.imread(img_name, cv2.IMREAD_ANYCOLOR)
            img_width = img.shape[1]
            img_height = img.shape[0]

            yolo_file = open(self.output_dir + file.split("/")[-1],"w+")
            with open(file, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    class_id = self.class_names[line[0]]
                    cx,cy,w,h = xyxy2xywh_np(line[4:8], img_width, img_height)
                    yolo_file.write(f"{class_id} {cx} {cy} {w} {h}\n")
            f.close()
            yolo_file.close()
            
            # with open(self.output_dir + file.split("\\")[-1],"w+") as yolo_file:
            #     yolo_file.write(f"{class_id} {cx} {cy} {w} {h}\n")

            # visualization
            # img_path = file.split('\\')[-1].split('.')[0]
            # img = self.img_train_dir + img_path + ".png"
            # if visualization: 
            #     eda(img, labels)

    def __len__(self):
        return len(self.label_dir_list)


if __name__ == "__main__":
    convert = convert2yolo()
    convert.save()