import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...utils import loss_utils
from ...ops.iou3d_nms import iou3d_nms_utils

def KLDivergenceLoss(y, teacher_scores, mask=None, T=1):
    if mask is not None:
        if mask.sum() > 0:
            p = F.log_softmax(y/T, dim=1)[mask]
            q = F.softmax(teacher_scores/T, dim=1)[mask]
            l_kl = F.kl_div(p, q, reduce=False)
            loss = torch.sum(l_kl)
            loss = loss / mask.sum()
        else:
            loss = torch.Tensor([0]).cuda()
    else:
        p = F.log_softmax(y/T, dim=1)
        q = F.softmax(teacher_scores/T, dim=1)
        l_kl = F.kl_div(p, q, reduce=False)
        loss = l_kl.sum() / l_kl.size(0)
    return loss * T**2

class AnchorHeadSingleSGM(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(input_channels, 1)

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
        self.add_module(
            'feature_loss_func',
            getattr(loss_utils, 'WeightedL2WithSigmaLoss')()
        )
        self.add_module(
            'object_score_loss_func',
            getattr(loss_utils, 'WeightedSmoothL1Loss')(code_weights=[1.0, 1.0, 1.0])
        )
        self.add_module(
            'object_dir_loss_func',
            nn.MSELoss()
        )
        
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        pool = self.avgpool(spatial_features_2d.detach())
        pool = torch.flatten(pool, 1)
        uncertainty = nn.Sigmoid()(self.fc(pool))
        uncertainty = uncertainty.view(-1).mean()

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['uncertainty'] = uncertainty

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            
            self.forward_ret_dict['distilled_batch_cls_preds'] = data_dict['distilled_batch_cls_preds']
            self.forward_ret_dict['distilled_batch_box_preds'] = data_dict['distilled_batch_box_preds']
            self.forward_ret_dict['distilled_batch_dir_cls_preds'] = data_dict['distilled_batch_dir_cls_preds']

            self.forward_ret_dict['distilled_spatial_features'] = data_dict['distilled_spatial_features']
            self.forward_ret_dict['spatial_features'] = data_dict['spatial_features']

            self.forward_ret_dict['distilled_spatial_features_2d'] = data_dict['distilled_spatial_features_2d']
            self.forward_ret_dict['spatial_features_2d'] = data_dict['spatial_features_2d']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
        
    def get_cls_layer_loss(self):
        distilled_batch_cls_preds = self.forward_ret_dict['distilled_batch_cls_preds'] # (B, num_boxes, num_classes)
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        pos_inds = one_hot_targets == 1
        anchor_loss_fg = self.model_cfg.ANCHOR_FG_WEIGHTS * KLDivergenceLoss(cls_preds, distilled_batch_cls_preds, pos_inds)
        pos_inds = one_hot_targets == 0
        anchor_loss_bg = self.model_cfg.ANCHOR_BG_WEIGHTS * KLDivergenceLoss(cls_preds, distilled_batch_cls_preds, pos_inds)
        anchor_loss =  anchor_loss_fg + anchor_loss_bg
        anchor_loss = anchor_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['anchor_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item(),
            'rpn_loss_anchor_fg': anchor_loss_fg.item(),
            'rpn_loss_anchor_bg': anchor_loss_bg.item(),
        }
        return cls_loss + anchor_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def nn_distance(self, box1, box2, iou_thres=0.8, return_loss='10'):
        """
            box1: (N, C) torch tensor;   box2: (M, C) torch tensor
        """
        ans_iou = iou3d_nms_utils.boxes_iou_bev(box1, box2)
        iou1, idx1 = torch.max(ans_iou, dim=1)
        iou2, idx2 = torch.max(ans_iou, dim=0)

        mask1, mask2 = iou1 > iou_thres, iou2 > iou_thres
        ans_iou = ans_iou[mask1]
        ans_iou = ans_iou[:, mask2]

        if ans_iou.shape[0] == 0 or ans_iou.shape[1] == 0:  # for unlabeled data (some scenes wo cars)
            return [None] * 5

        iou1, idx1 = torch.max(ans_iou, dim=1)
        iou2, idx2 = torch.max(ans_iou, dim=0)
        val_box1, val_box2 = box1[mask1], box2[mask2]
        aligned_box1, aligned_box2 = val_box1[idx2], val_box2[idx1]

        # iou3d, iou_bev = iou3d_utils.boxes_aligned_iou3d_gpu(val_box2, aligned_box1, need_bev=True)
        encoded_box_preds, encoded_reg_targets = self.add_sin_difference(val_box1.unsqueeze(0), aligned_box2.unsqueeze(0))
        loss1 = self.reg_loss_func(encoded_box_preds, encoded_reg_targets)
        encoded_box_preds, encoded_reg_targets = self.add_sin_difference(val_box2.unsqueeze(0), aligned_box1.unsqueeze(0))
        loss2 = self.reg_loss_func(encoded_box_preds, encoded_reg_targets)
        if return_loss == '10':
            box_object_loss = loss1.sum() / loss1.shape[0]
        elif return_loss == '01':
            box_object_loss = loss2.sum() / loss2.shape[0]
        elif return_loss == '11':
            box_object_loss = (loss1.sum() + loss2.sum()) / (loss1.shape[0] + loss2.shape[0])
        else:
            raise NotImplementedError

        return box_object_loss, idx1, idx2, mask1, mask2

    def get_object_loss(self):
        '''
            each prediction of student matched with one prediction of teacher
        '''
        cls_preds = self.forward_ret_dict['cls_preds']
        batch_size = int(cls_preds.shape[0])
        box_preds = self.forward_ret_dict['box_preds']
        dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)

        batch_cls_preds_stu, batch_box_preds_stu = self.generate_predicted_boxes(
                batch_size=batch_size,
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
        batch_dir_cls_preds_stu = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
        
        batch_cls_preds_tea = self.forward_ret_dict['distilled_batch_cls_preds'] 
        batch_box_preds_tea = self.forward_ret_dict['distilled_batch_box_preds']
        batch_dir_cls_preds_tea = self.forward_ret_dict['distilled_batch_dir_cls_preds']
        
        batch_cls_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_box_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_dir_loss = torch.tensor([0.], dtype=torch.float32).cuda()

        for cls_preds_stu, cls_preds_tea, box_preds_stu, box_preds_tea,  dir_cls_preds_stu, dir_cls_preds_tea, in \
            zip(batch_cls_preds_stu, batch_cls_preds_tea, batch_box_preds_stu, batch_box_preds_tea, batch_dir_cls_preds_stu, batch_dir_cls_preds_tea):
            
            cls_preds_top_stu = cls_preds_stu[torch.arange(0, cls_preds_stu.shape[0]), cls_preds_stu.argmax(1)]
            idx_stu = torch.gt(torch.sigmoid(cls_preds_top_stu), self.model_cfg.OBJECT_STU_SCORE)
            
            cls_preds_top_tea = cls_preds_tea[torch.arange(0, cls_preds_tea.shape[0]), cls_preds_tea.argmax(1)]
            idx_tea = torch.gt(torch.sigmoid(cls_preds_top_tea), self.model_cfg.OBJECT_TEA_SCORE)

            top_box_preds_stu, top_box_preds_tea, top_cls_preds_stu, top_cls_preds_tea, top_dir_preds_stu, top_dir_preds_tea \
                = box_preds_stu[idx_stu], box_preds_tea[idx_tea], cls_preds_stu[idx_stu], cls_preds_tea[idx_tea], dir_cls_preds_stu[idx_stu], dir_cls_preds_tea[idx_tea]

            if idx_stu.sum() > 0 and idx_tea.sum() > 0:

                iou_threshold = float(self.model_cfg.get('OBJECT_IOU_THRESHOLD', 0.8))
                # uncertainty = self.forward_ret_dict["uncertainty"]
                # iou_threshold = iou_threshold * torch.exp(-uncertainty) + uncertainty
                # iou_threshold = iou_threshold * 0.8
                # print(iou_threshold)

                # iou_threshold = 0.8
                
                # center object loss
                box_object_loss, idx1, idx2, mask1, mask2 = self.nn_distance(top_box_preds_stu, top_box_preds_tea, iou_threshold)
                if box_object_loss is None:
                    continue
                batch_box_loss += box_object_loss

                # cls_score object loss
                aligned_cls_preds_stu, aligned_cls_preds_tea = top_cls_preds_stu[mask1][idx2], top_cls_preds_tea[mask2][idx1]
                scores_stu, scores_tea = torch.sigmoid(top_cls_preds_stu[mask1]), torch.sigmoid(aligned_cls_preds_tea)
                score_object_loss = self.object_score_loss_func(scores_stu, scores_tea).mean()
                batch_cls_loss += score_object_loss

                # dir object loss
                aligned_dir_preds_tea = top_dir_preds_tea[mask2][idx1]
                aligned_dir_preds_tea = F.softmax(aligned_dir_preds_tea, dim=-1)
                top_dir_preds_stu = F.softmax(top_dir_preds_stu[mask1], dim=-1)
                dir_object_loss = self.object_dir_loss_func(top_dir_preds_stu, aligned_dir_preds_tea)
                batch_dir_loss += dir_object_loss

        loss_object_box = batch_box_loss / batch_size
        loss_object_cls = batch_cls_loss / batch_size
        loss_object_dir = batch_dir_loss / batch_size
        object_loss = loss_object_box + loss_object_cls + loss_object_dir
        object_loss = object_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['object_weight']
        
        tb_dict = {
            'loss_object_box': loss_object_box.item(),
            'loss_object_cls': loss_object_cls.item(),
            'loss_object_dir': loss_object_dir.item(),
            'object_loss': object_loss.item(),
        }
        
        return object_loss, tb_dict
    
    def get_feature_loss(self):
        features_preds = self.forward_ret_dict['spatial_features_2d']
        features_targets = self.forward_ret_dict['distilled_spatial_features_2d']
        batch_size = int(features_preds.shape[0])

        features_preds = features_preds.permute(0, *range(2, len(features_preds.shape)), 1) # B, H, W, C
        features_targets = features_targets.permute(0, *range(2, len(features_targets.shape)), 1)

        positives = self.forward_ret_dict["box_cls_labels"] > 0
        positives = positives.view(*features_preds.shape[:-1], self.num_anchors_per_location)
        positives = torch.any(positives, dim=-1)
        
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=10)

        pos_inds = reg_weights > 0
        pos_feature_preds = features_preds[pos_inds]
        pos_feature_targets = features_targets[pos_inds]
        pos_reg_weights = reg_weights[pos_inds]
        feature_loss_src = self.feature_loss_func(pos_feature_preds,
                                                      pos_feature_targets,
                                                      pos_reg_weights)
        feature_loss_src = feature_loss_src.mean(-1)

        feature_loss = feature_loss_src.sum() / batch_size
        feature_loss = feature_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['feature_weight']

        tb_dict = {
            'rpn_loss_feature': feature_loss.item(),
        }

       
        return feature_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)

        rpn_loss = cls_loss + box_loss
        
        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['object_weight'] > 0.:
            object_loss, tb_dict_object = self.get_object_loss()
            rpn_loss = rpn_loss + object_loss
            tb_dict.update(tb_dict_object)
        
        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['feature_weight'] > 0.:
            feature_loss, tb_dict_feature = self.get_feature_loss()
            rpn_loss = rpn_loss + feature_loss
            tb_dict.update(tb_dict_feature)
            
        tb_dict['rpn_loss'] = rpn_loss.item()

        return rpn_loss, tb_dict
