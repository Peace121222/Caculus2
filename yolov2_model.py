import torch
import torch.nn as nn

class TinyDarknet(nn.Module):
    def __init__(self):
        super(TinyDarknet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        return x

class YOLOv2(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5, grid_size=13):
        super(YOLOv2, self).__init__()
        self.backbone = TinyDarknet()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Đầu ra: SxS ô, mỗi ô có num_anchors * (5 + num_classes) giá trị
        self.conv_out = nn.Conv2d(128, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_out(x)
        # Reshape: [batch, S, S, num_anchors * (5 + num_classes)]
        batch_size = x.size(0)
        x = x.view(batch_size, self.grid_size, self.grid_size, self.num_anchors, 5 + self.num_classes)
        return x

# Khởi tạo mô hình
model = YOLOv2(num_classes=20, num_anchors=5, grid_size=13)
def yolo_v2_loss(pred, target, anchors, lambda_coord=5.0, lambda_noobj=0.5):
    """
    pred: [batch, S, S, num_anchors, 5 + num_classes]
    target: [batch, S, S, num_anchors, 5 + num_classes]
    anchors: Danh sách [w, h] của anchor boxes
    """
    batch_size, S, _, num_anchors, _ = pred.shape
    device = pred.device
    anchors = torch.tensor(anchors, device=device)

    # Tách thành phần từ pred và target
    pred_xy = torch.sigmoid(pred[..., 0:2])  # Dự đoán x, y (sigmoid để giới hạn 0-1)
    pred_wh = pred[..., 2:4]  # Dự đoán w, h
    pred_conf = torch.sigmoid(pred[..., 4])  # Confidence
    pred_cls = pred[..., 5:]  # Xác suất lớp

    target_xy = target[..., 0:2]
    target_wh = target[..., 2:4]
    target_conf = target[..., 4]
    target_cls = target[..., 5:]

    # Tính IoU để xác định responsible anchor
    pred_boxes = torch.cat([pred_xy, torch.exp(pred_wh) * anchors.view(1, 1, 1, num_anchors, 2)], dim=-1)
    target_boxes = torch.cat([target_xy, target_wh], dim=-1)
    iou = compute_iou(pred_boxes, target_boxes)  # [batch, S, S, num_anchors]

    # Mask cho object và no-object
    obj_mask = target_conf > 0  # Ô có object
    noobj_mask = target_conf == 0  # Ô không có object
    responsible_mask = iou == iou.max(dim=-1, keepdim=True)[0]  # Anchor có IoU cao nhất

    # 1. Localization Loss (chỉ tính cho responsible anchor có object)
    coord_loss = lambda_coord * torch.sum(
        obj_mask * responsible_mask * (
            (pred_xy - target_xy)**2 + (torch.log(torch.exp(pred_wh) + 1e-6) - torch.log(target_wh + 1e-6))**2
        )
    )

    # 2. Confidence Loss
    conf_loss_obj = torch.sum(obj_mask * responsible_mask * (pred_conf - iou)**2)
    conf_loss_noobj = lambda_noobj * torch.sum(noobj_mask * pred_conf**2)

    # 3. Classification Loss
    class_loss = torch.sum(obj_mask * responsible_mask * torch.sum((pred_cls - target_cls)**2, dim=-1))

    total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
    return total_loss

def compute_iou(pred_boxes, target_boxes):
    """Tính IoU giữa predicted và target boxes"""
    pred_xy, pred_wh = pred_boxes[..., :2], pred_boxes[..., 2:4]
    target_xy, target_wh = target_boxes[..., :2], target_boxes[..., 2:4]

    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    intersect_wh = torch.min(pred_wh, target_wh).clamp(min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    union_area = pred_area + target_area - intersect_area
    return intersect_area / (union_area + 1e-6)