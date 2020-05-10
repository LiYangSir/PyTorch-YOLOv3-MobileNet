from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from InvertedResidual import InvertedResidual, extend_layers, output_layers, conv_dbl, con1x1
from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv0 = conv_dbl(3, 32, 2)  # First DownSample 416 -> 208
        self.trunk52 = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),  # Second DownSample 208 -> 104
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),  # Third DownSample 104 -> 52
            InvertedResidual(32, 32, 1, 6),
        )
        self.trunk26 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),  # Fourth DownSample 52 -> 26
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
        )
        self.trunk13 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 160, 2, 6),  # Fifth DownSample 26 -> 13
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
        )

        self.conEt1 = extend_layers(160, 512)
        self.conOp1 = output_layers(512, 255)
        self.conUp1 = nn.Sequential(con1x1(512, 256), nn.ConvTranspose2d(256, 256, 3, 2, 1, 1, 256))

        self.conEt2 = extend_layers(320, 256)
        self.conOp2 = output_layers(256, 255)
        self.conUp2 = nn.Sequential(con1x1(256, 128), nn.ConvTranspose2d(128, 128, 3, 2, 1, 1, 128))

        self.conEt3 = extend_layers(160, 256)
        self.conOp3 = output_layers(256, 255)

        self.yolo13 = YOLOLayer([(116, 90), (156, 198), (373, 326)], 80, 416)
        self.yolo26 = YOLOLayer([(30, 61), (62, 45), (59, 119)], 80, 416)
        self.yolo52 = YOLOLayer([(10, 13), (16, 30), (33, 23)], 80, 416)

    def forward(self, x, target=None):
        img_dim = x.shape[2]
        x = self.conv0(x)
        x = self.trunk52(x)
        xR52 = x
        x = self.trunk26(x)
        xR26 = x
        x = self.trunk13(x)
        x = self.conEt1(x)
        xOp13 = self.conOp1(x)
        x = self.conUp1(x)
        x = torch.cat([x, xR26], 1)

        x = self.conEt2(x)
        xOp26 = self.conOp2(x)
        x = self.conUp2(x)
        x = torch.cat([x, xR52], 1)
        x = self.conEt3(x)
        xOp52 = self.conOp3(x)

        out13, loss13 = self.yolo13(xOp13, target, img_dim)
        out26, loss26 = self.yolo26(xOp26, target, img_dim)
        out52, loss52 = self.yolo52(xOp52, target, img_dim)

        return [out13, out26, out52], loss13 + loss26 + loss52

    def load_pretrained_params(self):
        deviceBool = next(self.parameters()).is_cuda
        device = torch.device("cuda:0" if deviceBool else "cpu")
        try:
            pretrainedDict = torch.load(self._opt.pretrainedParamFile, map_location=device.type)
            modelDict = self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except:
            print("Can't load pre-trained parameter files")


if __name__ == '__main__':
    input = torch.randn(1, 3, 416, 416)
    model = MobileNet()
    output = model(input)
    print()


