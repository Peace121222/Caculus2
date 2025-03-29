import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import xml.etree.ElementTree as ET  # Để đọc file XML nếu dùng VOC
from torch.utils.data import Dataset, DataLoader

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # YOLOv2 dùng 416x416
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Hàm đọc nhãn từ file XML (VOC format)
def parse_voc_annotation(xml_path, img_width, img_height, grid_size=13, num_classes=20):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    classes = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        # Giả định danh sách lớp: cần ánh xạ tên lớp thành chỉ số (index)
        class_dict = {"person": 0, "car": 1, "dog": 2}  # Ví dụ, thay bằng danh sách lớp của bạn
        cls_id = class_dict.get(class_name, 0)  # Mặc định là 0 nếu không tìm thấy

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Chuẩn hóa tọa độ về [0, 1]
        x = (xmin + xmax) / 2.0 / img_width
        y = (ymin + ymax) / 2.0 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        boxes.append([x, y, w, h])
        classes.append(cls_id)

    return boxes, classes

# Hàm tạo tensor nhãn cho YOLOv2
def create_target_tensor(boxes, classes, anchors, grid_size=13, num_classes=20):
    target = torch.zeros(grid_size, grid_size, len(anchors), 5 + num_classes)
    for box, cls in zip(boxes, classes):
        x, y, w, h = box
        grid_x, grid_y = int(x * grid_size), int(y * grid_size)
        cell_x, cell_y = x * grid_size - grid_x, y * grid_size - grid_y

        # Tìm anchor gần nhất
        iou_with_anchors = compute_iou_anchor(torch.tensor([w, h]), torch.tensor(anchors))
        anchor_idx = iou_with_anchors.argmax()

        # Gán giá trị vào tensor
        target[grid_y, grid_x, anchor_idx, 0:5] = torch.tensor([cell_x, cell_y, w, h, 1.0])
        target[grid_y, grid_x, anchor_idx, 5 + cls] = 1.0
    return target

# Hàm tính IoU giữa box và anchors
def compute_iou_anchor(box, anchors):
    box_area = box[0] * box[1]
    anchors_area = anchors[:, 0] * anchors[:, 1]
    intersect_wh = torch.min(box, anchors)
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]
    union_area = box_area + anchors_area - intersect_area
    return intersect_area / (union_area + 1e-6)

# Dataset tùy chỉnh cho YOLOv2
class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, anchors, grid_size=13, num_classes=20):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.anchors = anchors
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.xml').replace('.png', '.xml'))

        # Tải và xử lý ảnh
        image = preprocess_image(img_path)

        # Tải kích thước ảnh gốc
        img_pil = Image.open(img_path)
        img_width, img_height = img_pil.size

        # Tải nhãn từ file XML
        boxes, classes = parse_voc_annotation(label_path, img_width, img_height, self.grid_size, self.num_classes)

        # Tạo tensor nhãn
        target = create_target_tensor(boxes, classes, self.anchors, self.grid_size, self.num_classes)

        return image, target

# Ví dụ sử dụng
if __name__ == "__main__":
    anchors = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
    image_dir = "/path/to/your/images"  # Thay bằng đường dẫn tới thư mục ảnh
    label_dir = "/path/to/your/labels"  # Thay bằng đường dẫn tới thư mục nhãn XML

    dataset = YOLODataset(image_dir, label_dir, anchors)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Kiểm tra một batch
    for images, targets in dataloader:
        print("Images shape:", images.shape)  # torch.Size([4, 3, 416, 416])
        print("Targets shape:", targets.shape)  # torch.Size([4, 13, 13, 5, 25])
        break