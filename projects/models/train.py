import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class TrafficYoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Load label
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        boxes = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip() == "":
                        continue
                    class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())

                    # Convert YOLO format to [x_min, y_min, x_max, y_max]
                    x_center *= width
                    y_center *= height
                    box_w *= width
                    box_h *= height

                    x_min = x_center - box_w / 2
                    y_min = y_center - box_h / 2
                    x_max = x_center + box_w / 2
                    y_max = y_center + box_h / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    class_labels.append(int(class_id))

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_labels = torch.tensor(class_labels, dtype=torch.long)

        # Apply image transforms if any
        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': class_labels,
            'image_id': torch.tensor([idx])
        }

        return image, target
    

from torchvision.transforms import ToTensor, Resize, Compose

transform = Compose([
    Resize((128, 128)),
    ToTensor()
])

train_dataset = TrafficYoloDataset(
    image_dir="D:/STUDYYY/vehicle_detection/project/dataset/traffic_data/train/images",
    label_dir="D:/STUDYYY/vehicle_detection/project/dataset/traffic_data/train/labels",
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
