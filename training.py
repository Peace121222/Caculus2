import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
from torch.utils.data import Dataset, DataLoader

# Định nghĩa backbone TinyDarknet
class TinyDarknet(nn.Module):
    def __init__(self):
        super(TinyDarknet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        return x

# Định nghĩa mô hình YOLOv2 cho 1 lớp (water stain)
class YOLOv2(nn.Module):
    def __init__(self, num_classes=1, num_anchors=5, grid_size=13):
        super(YOLOv2, self).__init__()
        self.backbone = TinyDarknet()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv_out = nn.Conv2d(256, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_out(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.grid_size, self.grid_size, self.num_anchors, 5 + self.num_classes)
        return x

# Hàm tính IoU
def compute_iou(pred_boxes, target_boxes):
    pred_xy, pred_wh = pred_boxes[..., :2], pred_boxes[..., 2:4]
    target_xy, target_wh = target_boxes[..., :2], target_boxes[..., 2:4]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    intersect_wh = torch.min(pred_wh, target_wh).clamp(min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    union_area = pred_area + target_area - intersect_area
    return intersect_area / (union_area + 1e-6)

# Hàm Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.3):  # Giảm ngưỡng IoU để lọc box chặt hơn
    if not boxes:
        return [], [], []
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    classes = torch.tensor([0] * len(boxes))

    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0].item()
        keep.append(i)

        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter)

        mask = iou < iou_threshold
        if mask.sum() == 0:
            break
        order = order[1:][mask]

    kept_boxes = boxes[keep]
    kept_boxes = torch.stack([
        (kept_boxes[:, 0] + kept_boxes[:, 2]) / 2,
        (kept_boxes[:, 1] + kept_boxes[:, 3]) / 2,
        kept_boxes[:, 2] - kept_boxes[:, 0],
        kept_boxes[:, 3] - kept_boxes[:, 1]
    ], dim=1).tolist()
    kept_scores = scores[keep].tolist()
    kept_classes = classes[keep].tolist()
    return kept_boxes, kept_scores, kept_classes

# Hàm loss của YOLOv2
def yolo_v2_loss(pred, target, anchors, lambda_coord=5.0, lambda_noobj=0.5):
    batch_size, S, _, num_anchors, _ = pred.shape
    device = pred.device
    anchors = torch.tensor(anchors, device=device)

    pred_xy = torch.sigmoid(pred[..., 0:2])
    pred_wh = pred[..., 2:4]
    pred_conf = torch.sigmoid(pred[..., 4])
    pred_cls = pred[..., 5:]

    target_xy = target[..., 0:2]
    target_wh = target[..., 2:4]
    target_conf = target[..., 4]
    target_cls = target[..., 5:]

    pred_boxes = torch.cat([pred_xy, torch.exp(pred_wh) * anchors.view(1, 1, 1, num_anchors, 2)], dim=-1)
    target_boxes = torch.cat([target_xy, target_wh], dim=-1)
    iou = compute_iou(pred_boxes, target_boxes)

    obj_mask = target_conf > 0
    noobj_mask = target_conf == 0
    responsible_mask = iou == iou.max(dim=-1, keepdim=True)[0]

    obj_mask = obj_mask.unsqueeze(-1)
    responsible_mask = responsible_mask.unsqueeze(-1)

    coord_loss = lambda_coord * torch.sum(
        obj_mask * responsible_mask * (
            (pred_xy - target_xy)**2 + (torch.log(torch.exp(pred_wh) + 1e-6) - torch.log(target_wh + 1e-6))**2
        )
    )
    conf_loss_obj = torch.sum(obj_mask.squeeze(-1) * responsible_mask.squeeze(-1) * (pred_conf - iou)**2)
    conf_loss_noobj = lambda_noobj * torch.sum(noobj_mask * pred_conf**2)
    class_loss = torch.sum(obj_mask.squeeze(-1) * responsible_mask.squeeze(-1) * torch.sum((pred_cls - target_cls)**2, dim=-1))

    total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
    return total_loss

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Hàm tạo nhãn giả lập cho vết nước
def create_sample_target(anchors, grid_size=13, num_classes=1):
    target = torch.zeros(grid_size, grid_size, len(anchors), 5 + num_classes)
    # Dựa trên hình mẫu, vết nước nằm ở ô (5, 6)
    target[5, 6, 0, 0:5] = torch.tensor([0.5, 0.2, 0.05, 0.3, 1.0])  # x, y, w, h, confidence
    target[5, 6, 0, 5] = 1.0  # Lớp 0 (water stain)
    return target

# Dataset tùy chỉnh cho folder data
class ImageDataset(Dataset):
    def __init__(self, data_dir, anchors, grid_size=13, num_classes=1):
        self.data_dir = data_dir
        self.anchors = anchors
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = preprocess_image(img_path)
        target = create_sample_target(self.anchors, self.grid_size, self.num_classes)
        return image, target

# Hàm vẽ bounding box lên ảnh
def draw_bounding_box(image_path, boxes, classes, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    for box, cls in zip(boxes, classes):
        x, y, w, h = box
        x_center = x * img_width
        y_center = y * img_height
        box_width = w * img_width
        box_height = h * img_height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2

        draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
        draw.text((x_min, y_min - 10), "Water Stain", fill="blue")

    image.save(output_path)
    print(f"Ảnh kết quả đã được lưu tại: {output_path}")

# Huấn luyện trên toàn bộ folder data
anchors = [[0.05, 0.3], [0.03, 0.2], [0.07, 0.4], [0.04, 0.25], [0.06, 0.35]]  # Anchors mới
model = YOLOv2(num_classes=1, num_anchors=5, grid_size=13)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Giảm learning rate để học ổn định hơn

# Đường dẫn tới folder data
data_dir = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Folder {data_dir} không tồn tại. Hãy tạo folder 'data' và thêm ảnh vào.")

# Tạo dataset và dataloader
dataset = ImageDataset(data_dir, anchors)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Huấn luyện
num_epochs = 150  # Tăng số epoch để học tốt hơn
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = yolo_v2_loss(outputs, targets, anchors)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

# Dự đoán và vẽ bounding box cho tất cả ảnh trong folder
model.eval()
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for img_name in dataset.image_files:
        img_path = os.path.join(data_dir, img_name)
        image = preprocess_image(img_path).unsqueeze(0)
        output = model(image)
        
        boxes, scores, classes = [], [], []
        for i in range(13):
            for j in range(13):
                for a in range(5):
                    pred = output[0, i, j, a]
                    conf = torch.sigmoid(pred[4])
                    if conf > 0.6:  # Tăng ngưỡng để chỉ giữ dự đoán chắc chắn
                        x = (torch.sigmoid(pred[0]) + j) / 13
                        y = (torch.sigmoid(pred[1]) + i) / 13
                        w = torch.exp(pred[2]) * anchors[a][0] / 13
                        h = torch.exp(pred[3]) * anchors[a][1] / 13
                        boxes.append([x.item(), y.item(), w.item(), h.item()])
                        scores.append(conf.item())
                        cls_probs = torch.softmax(pred[5:], dim=0)
                        classes.append(cls_probs.argmax().item())
        
        # Áp dụng NMS để lọc box chồng lấn
        boxes, scores, classes = nms(boxes, scores, iou_threshold=0.3)

        print(f"\nImage: {img_name}")
        print("Detected boxes:", boxes)
        print("Scores:", scores)
        print("Classes:", classes)

        if boxes:
            output_path = os.path.join(output_dir, f"output_{img_name}")
            draw_bounding_box(img_path, boxes, classes, output_path)
        else:
            print("Không phát hiện vết nước nào với confidence > 0.6")