from albumentations import Compose, Rotate, RandomCrop, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
import cv2

def augment_image(image, bbox):
    transform = Compose([
        Rotate(limit=15, p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomCrop(height=512, width=512, p=0.5),
        ToTensorV2(),
    ])
    augmented = transform(image=image, bboxes=[bbox], class_labels=['class_name'])
    return augmented['image'], augmented['bboxes']
