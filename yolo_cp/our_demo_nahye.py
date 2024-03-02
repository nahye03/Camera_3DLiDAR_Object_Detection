"""
python our_demo_nahye2.py --cfg_file /workspace/yolo_cp/tools/cfgs/kitti_models/centerpoint.yaml --ckpt checkpoint_epoch_80.pth --data_path /workspace/KITTI/velodyne/train --weight best.pt --conf 0.25 --img-size 640 --source /workspace/KITTI/images/train --nosave

python our_demo_nahye2.py --cfg_file /workspace/yolo_cp/tools/cfgs/kitti_models/centerpoint_yolo.yaml --ckpt checkpoint_epoch_80_yolodata.pth --data_path /workspace/KITTI/velodyne/train --weight best.pt --conf 0.25 --img-size 640 --source /workspace/KITTI/images/train --nosave > /workspace/yolo_cp/nahye_output/nohup_yolo.out

- box fusion
python our_demo_nahye2.py --cfg_file /workspace/yolo_cp/tools/cfgs/kitti_models/centerpoint_yolo.yaml --ckpt checkpoint_epoch_80_yolodata.pth --data_path /workspace/KITTI/velodyne/train --weight best.pt --conf 0.25 --img-size 640 --source /workspace/KITTI/images/train --nosave --box_fusion

-nohup
nohup python our_demo_nahye2.py --cfg_file /workspace/yolo_cp/tools/cfgs/kitti_models/centerpoint_yolo.yaml --ckpt checkpoint_epoch_80_yolodata.pth --data_path /workspace/KITTI/velodyne/train --weight best.pt --conf 0.25 --img-size 640 --source /workspace/KITTI/images/train --nosave --box_fusion > /workspace/yolo_cp/nahye_output/nohup_box_fusion2.out 2>&1 &

"""

# import warnings
# warnings.filterwarnings("ignore")

import time
import glob
import math
import argparse
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import seaborn as sns
import scipy.optimize
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RANSACRegressor
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict

# for yolov7
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# for centerpoint
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        angles = np.arctan2(points[:, 1], points[:, 0]) * (180 / np.pi)

        # set sight angle (-45 degree to 45 degree)
        min_angle = -45
        max_angle = 45

        # point cloud filtering
        filtered_data = points[(angles >= min_angle) & (angles <= max_angle)]
        points = filtered_data

        #seperate road and object point
        road_points, object_points = self.separate_road_and_objects(points)

        input_dict = {
            # 'road_points':road_points,
            'points': object_points,
            'frame_id': index,
        }


        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def separate_road_and_objects(self, points):
        #road model estimate using RANSAC
        ransac = RANSACRegressor()
        ransac.fit(points[:, :2], points[:, 2])  # using x, y and z

        # remove outlier
        inlier_mask = ransac.inlier_mask_
        road_points = points[inlier_mask]

        # get object points except road
        object_points = points[~inlier_mask]

        return road_points, object_points

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') 
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='nahye', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--box_fusion', action='store_true', help='evaluate box fusion method')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

class YOLOv7Detector:
    def __init__(self, args):
        self.args = args

        # Initialize
        set_logging()
        self.device = select_device(args.device)
        self.half = self.device.type != 'cpu'

        # Load model
        self.yolo_model = self._load_model(args.weights)

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        self.dataset = self._set_dataloader(args.source, args.img_size)

        # Get names and colors
        self.names = self.yolo_model.module.names if hasattr(self.yolo_model, 'module') else self.yolo_model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if (not args.nosave and not args.source.endswith('.txt')): #save image, txt
            # Directories
            self.save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))  # increment run
            (self.save_dir / 'labels' if args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    def __getitem__(self, index):
        return self.dataset[index]

    def detect(self, img):
        # pre_process image
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inference
        with torch.no_grad():
            pred = self.yolo_model(img, augment=self.args.augment)[0]

        # nms
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)

        return pred, img

    def save_result(self, pred, path, img, im0, vid_cap):
        #pred type : tensor (not list)
        if not len(pred):
            return

        path = Path(path)  # to Path
        frame = getattr(self.dataset, 'frame', 0)
        save_path = str(self.save_dir / path.name)  # ex) save_path :  runs/detect/nahye8/000015.png
        txt_path = str(self.save_dir / 'labels' / path.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # ex)txt_path : runs/detect/nahye8/labels/000015

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(pred):
            if args.save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

        # Save results (image with detections)
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
            print(f" The image with the result is saved in: {save_path}")
        else:  # 'video' or 'stream'
            if self.vid_path != save_path:  # new video
                self.vid_path = save_path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer.write(im0)

        return im0


    def _load_model(self, weights):
        model = attempt_load(weights, map_location=self.device)
        if not args.no_trace: #true
            model = TracedModel(model, self.device, args.img_size) # pytorch model transformation

        if self.half:
            model.half()  # to FP16

        return model

    def _set_dataloader(self, source, imgsz):
        stride = int(self.yolo_model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        return dataset

class CenterPointDetector:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.logger = common_utils.create_logger()
        self.logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

        # load dataset
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=self.logger
        )
        self.logger.info(f'Total number of samples: \t{len(self.demo_dataset)}')


        self.cp_model = self._build_model()

    def _build_model(self):
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=self.logger, to_cpu=True)
        model.cuda()
        model.eval()

        return model

    def detect(self, data):
        with torch.no_grad():
            data_dict = self.demo_dataset.collate_batch([data])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.cp_model.forward(data_dict)

        return pred_dicts

def read_calib_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    P2 = np.array([float(i) for i in lines[2].split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(i) for i in lines[4].split(' ')[1:]]).reshape(3,3)
    Tr_velo_to_cam = np.array([float(i) for i in lines[5].split(' ')[1:]]).reshape(3,4)

    R0 = np.eye(4)
    R0[:3, :3] = R0_rect
    Tr = np.vstack([Tr_velo_to_cam, [0,0,0,1]])

    return P2, R0, Tr

def calc_head(radians_corrected):
    degrees_corrected = radians_corrected * (180 / 3.141592)

    # Front: -45 degree ~ 45 degree
    if -45 < degrees_corrected <= 45:
        heading =  "front"
    # Right: 45 degree ~ 135 degree
    elif 45 < degrees_corrected <= 135:
        heading =  "left"
    # Back: 135 degree ~ 180 degree or -180 degree ~ -135 degree
    elif degrees_corrected > 135 or degrees_corrected <= -135:
        heading =  "back"
    # Left: -135 degree ~ -45 degree
    elif -135 < degrees_corrected <= -45:
        heading =  "right"

    return heading

def lidar_box3d_to_corner3d(bbox):
    '''
    bbox : [x, y, z, length(x), width(y), height(z), ry(yaw)]
            z
            |
        7 -------- 4
       /|         /|
      3 -------- 0 .
      | |        | |
      . 6 -------- 5
      |/         |/ ---y
      2 -------- 1
          /
         x
    '''
    x, y, z, l, w, h, ry = bbox
    corners = np.array([[l/2, w/2, h/2],
                        [l/2, w/2, -h/2],
                        [l/2, -w/2, -h/2],
                        [l/2, -w/2, h/2],
                        [-l/2, w/2, h/2],
                        [-l/2, w/2, -h/2],
                        [-l/2, -w/2, -h/2],
                        [-l/2, -w/2, h/2]])

    rotation_mat = np.array([[np.cos(ry), -np.sin(ry), 0],
                             [np.sin(ry),  np.cos(ry), 0],
                             [0,           0,          1]])

    rotated_corners = np.dot(rotation_mat, corners.T).T # (8,3)

    lidar_corners = rotated_corners + np.array([x, y, z])

    return lidar_corners

def lidar3d_to_image2d_projection(lidar_corners, P2, R0, Tr):
    '''
    lidar_corners : (8, 3)
    P2 : (3, 4)
    R0_rect : (4, 4)
    Tr_velo_to_cam : (4, 4)
    '''

    #transform lidar_corners shape : (8, 3) -> (8, 4)
    XYZ1 = np.hstack((lidar_corners, np.ones((lidar_corners.shape[0], 1))))

    # projection
    xyz = np.dot(P2,np.dot(R0,np.dot(Tr, XYZ1.T))) # (3, 8)
    z = xyz[2, :]
    x = np.where(z > 0, (xyz[0, :] / z).astype(np.int32), -(xyz[0, :] / z).astype(np.int32))
    y = np.where(z > 0, (xyz[1, :] / z).astype(np.int32), -(xyz[1, :] / z).astype(np.int32))

    return x, y

def corners_to_2dbox(cp_box_x, cp_box_y):

    x_min = np.min(cp_box_x)
    x_max = np.max(cp_box_x)
    y_min = np.min(cp_box_y)
    y_max = np.max(cp_box_y)

    return [x_min, y_min, x_max, y_max]

def match_bboxes(bbox_cp, bbox_yolo, IOU_THRESH=0.3):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_cp, bbox_yolo : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.

    Returns
    -------
    (idxs_cp, idxs_yolo, ious, labels)
        idxs_cp, idxs_yolo : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_cp = bbox_cp.shape[0]
    n_yolo = bbox_yolo.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_cp, n_yolo))
    for i in range(n_cp):
        for j in range(n_yolo):
            iou_matrix[i, j] = bbox_iou(bbox_cp[i,:], bbox_yolo[j,:])

    if n_yolo > n_cp:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_yolo - n_cp
      iou_matrix = np.concatenate( (iou_matrix,
                                    np.full((diff, n_yolo), MIN_IOU)),
                                  axis=0)

    if n_cp > n_yolo:
      # more ground-truth than predictions - add dummy columns
      diff = n_cp - n_yolo
      iou_matrix = np.concatenate( (iou_matrix,
                                    np.full((n_cp, diff), MIN_IOU)),
                                  axis=1)

    # call the Hungarian matching
    idxs_cp, idxs_yolo = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_cp.size) or (not idxs_yolo.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_cp, idxs_yolo]

    # remove dummy assignments
    sel_yolo = idxs_yolo < n_yolo
    idx_yolo_actual = idxs_yolo[sel_yolo]
    idx_cp_actual = idxs_cp[sel_yolo]
    ious_actual = iou_matrix[idx_cp_actual, idx_yolo_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_cp_actual[sel_valid], idx_yolo_actual[sel_valid]

def visualization_projection(img, x, y, color):
    img_h, img_w = img.shape[:2]

    for i, (ix, iy) in enumerate(zip(x, y)):
        if 0 <= ix < img_w and 0 <= iy < img_h:
            cv2.circle(img, (ix, iy), radius=1, color=color, thickness=2)

def visualization_3d_bbox(img, x, y, color):
    for i in range(4):
        cv2.line(img, (int(x[i]), int(y[i])),
                (int(x[(i + 1) % 4]), int(y[(i + 1) % 4])), color, 2)
    for i in range(4):
        cv2.line(img, (int(x[i]), int(y[i])),
                (int(x[i + 4]), int(y[i + 4])), color, 2)
    for i in range(4):
        cv2.line(img, (int(x[i + 4]), int(y[i + 4])),
                (int(x[(i + 1) % 4 + 4]), int(y[(i + 1) % 4 + 4])), color, 2)

def box_fusion_minmax(box1, box2):
    min_x = min(box1[0], box2[0])
    min_y = min(box1[1], box2[1])
    max_x = max(box1[2], box2[2])
    max_y = max(box1[3], box2[3])

    return [min_x, min_y, max_x, max_y]

def box_fusion_conf(box1, box2, box1_conf, box2_conf):
    new_box = []
    if box1_conf > box2_conf:
        new_box = box1
    else:
        new_box = box2

    return new_box

def weighted_box_fusion(box1, box2, box1_conf, box2_conf):
    total_conf = box1_conf + box2_conf
    min_x = (box1[0] * box1_conf + box2[0] * box2_conf) / total_conf
    min_y = (box1[1] * box1_conf + box2[1] * box2_conf) / total_conf
    max_x = (box1[2] * box1_conf + box2[2] * box2_conf) / total_conf
    max_y = (box1[3] * box1_conf + box2[3] * box2_conf) / total_conf

    return [min_x, min_y, max_x, max_y]

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def evaluate_box_fusion(file_list, pred_box, gt_box, iou_threshold=0.5):
    '''
    pred_box = (file name, label, bbox, conf)
    gt_box = (file name, label, bbox)
    '''
    total_ciou = 0

    for f in file_list:
        pred = [pred for pred in pred_box if pred[0] == f]
        gt = [gt for gt in gt_box if gt[0] == f]

        pred = sorted(pred, key = lambda conf : conf[-1], reverse=True)

        match_list = [-1] * len(gt)
        ciou_per_file = 0

        for i in range(len(gt)):
            iou_max = 0
            iou_idx = -1
            for j in range(len(pred)):
                if gt[i][1] == pred[j][1] and j not in match_list:
                    iou = bbox_iou(pred[j][2], gt[i][2], CIoU=True)
                    if iou > iou_max:
                        iou_max = iou
                        iou_idx = j

            if iou_max > iou_threshold:
                match_list[i] = iou_idx
                ciou_per_file += iou_max

        total_ciou += ciou_per_file

        print("file : ", f, " >> ciou : ", ciou_per_file/len(gt))

    return total_ciou / len(pred_box)

def calculate_ap(rec, prec):
    '''
    prec |
         |---
         |   \____
         |        \_
          -----------|
             recall
    '''
    ap = None
    if len(rec):
        mrec = [0.0] + [r for r in rec] + [1.0]
        mprec = [1.0] + [p for p in prec] + [0.0]

        for i in range(len(mprec) - 1, 0, -1):
            mprec[i-1] = max(mprec[i-1], mprec[i])

        change_points = []
        for i in range(len(mrec)-1):
            if mrec[i] != mrec[i+1]:
                change_points.append(i+1)

        ap = 0
        for point in change_points:
            ap = ap + np.sum((mrec[point] - mrec[point-1]) * mprec[point])

    return ap

def mAP(detections, groundtruths, classes, iou_threshold = 0.5):
    '''
    precision = TP / (TP + FP)
    recall = TP / (TP + FN) = TP / number of ground truth box


    detections = [f'{path[-10:-4]}', label_match, (min_x, min_y, max_x, max_y), conf_match]
    groundtruths = [f'{path[-10:-4]}', label_class, (np.float32(line[4]), np.float32(line[5]), np.float32(line[6]), np.float32(line[7]))]) # file name, label_class, (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
    '''

    ap_per_class = [] # ap

    for c in classes:
        detects = [d for d in detections if d[1] == c]
        gts = [g for g in groundtruths if g[1] == c]

        # total number of ground truth box
        npos = len(gts)

        detects = sorted(detects, key = lambda conf : conf[-1], reverse=True)

        TP = np.zeros(len(detects))
        FP = np.zeros(len(detects))

        # number of gt box for each image
        det = Counter(cc[0] for cc in gts) #Counter({'000001': 1})

        for key, val in det.items():
            det[key] = np.zeros(val)


        # total detected box
        for d in range(len(detects)):
            gt = [gt for gt in gts if gt[0] == detects[d][0]] # same file

            iou_max = 0

            # get iou for all gt box with detected box in same image
            for j in range(len(gt)):
                iou = bbox_iou(detects[d][2], gt[j][2])
                if iou > iou_max:
                    iou_max = iou
                    iou_idx = j

            if iou_max >= iou_threshold:
                if det[detects[d][0]][iou_idx] == 0 :
                    TP[d] = 1
                    det[detects[d][0]][iou_idx] = 1
                else :
                    FP[d] = 1
            else:
                FP[d] = 1

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        rec = acc_TP / npos
        prec = acc_TP / (acc_FP + acc_TP)

        ap = calculate_ap(rec, prec)
        ap_per_class.append(ap)
        print("class : ", c, " >>" , ap)

    mAP = np.mean([x for x in ap_per_class if x != None])
    print("mAP : ", mAP)
    return mAP

def make_confusion_matrix(file_list, pred_box, gt_box, iou_threshold = 0.5):
    '''
    pred_box = (file name, label, bbox, conf)
    gt_box = (file name, label, bbox)
    '''
    matrix_class = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'background']
    total_pred = []
    total_true = []

    tps = defaultdict(lambda: 0, {class_name: 0 for class_name in matrix_class})
    fps = defaultdict(lambda: 0, {class_name: 0 for class_name in matrix_class})
    fns = defaultdict(lambda: 0, {class_name: 0 for class_name in matrix_class})

    #file
    for f in file_list:
        print(f'evaluation {f}')
        pred = [pred for pred in pred_box if (pred[0] == f and pred[1] != 'DontCare')]
        gt = [gt for gt in gt_box if (gt[0] == f and gt[1] != 'DontCare')]

        pred = sorted(pred, key = lambda conf : conf[-1], reverse=True)

        match_list = []

        for i in range(len(gt)):
            best_iou = 0
            best_pred_idx = None
            best_pred_class = None

            for j in range(len(pred)):
                if j not in match_list:
                    iou = bbox_iou(pred[j][2], gt[i][2])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = j
                        best_pred_class = pred[j][1]

            if best_iou >= iou_threshold:
                if best_pred_class == gt[i][1]: # TP
                    total_pred.append(best_pred_class)
                    total_true.append(gt[i][1])

                    match_list.append(best_pred_idx)
                    tps[gt[i][1]] += 1
                else:
                    total_pred.append(best_pred_class)
                    total_true.append(gt[i][1])

                    fns[gt[i][1]] += 1

            else: #FN
                total_pred.append('background')
                total_true.append(gt[i][1])

                fns[gt[i][1]] += 1

        for i in range(len(pred)):
            if i not in match_list: #FP
                total_pred.append(pred[i][1])
                total_true.append('background')

                fps[pred[i][1]] += 1

    ap_per_class = []
    for class_id in [cls_id for cls_id in tps if cls_id != 'background']:
        tp = tps[class_id]
        fp = fps[class_id]
        fn = fns[class_id]

        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        ap = precision * recall
        ap_per_class.append(ap)
        print("class : ", class_id, " >>" , ap)

    mAP = np.mean(ap_per_class)
    print(f"mAP: {mAP}")

    # save confision matrix
    cm = confusion_matrix(total_true, total_pred, normalize ='true', labels=matrix_class)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=matrix_class, yticklabels=matrix_class)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.savefig('./nahye_output/confusion_matrix_final.png')

if __name__ == '__main__':

    args, cfg = parse_config()

    yolo_detector = YOLOv7Detector(args)
    cp_detector = CenterPointDetector(cfg, args)

    yolo_class = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    cp_class = ['', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in yolo_class]

    final_pred = [] # file name, label_class, (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax), conf
    gt_pred = [] # file name, label_class, (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
    file_list = []

    final_yolo = []
    final_cp = []

    if args.box_fusion:
        # file name, bbox, conf
        yolo_box = []
        cp_box = []
        cp_cloud_point_box = []
        minmax_box = []
        conf_box = []
        weighted_box = []

    data_list_path = '/workspace/KITTI/val_cp.txt'
    with open(data_list_path, 'r') as f:
        data_list = [x.split() for x in f.read().strip().splitlines()]

    for idx in data_list:
        idx = int(idx[0])
    # for idx in range(6749, 6750, 1):
        print("file number : ", idx)
        # yolo
        print("------------------yolo--------------------")
        path, img, final_img, vid_cap = yolo_detector.dataset[idx]
        file_list.append(f'{path[-10:-4]}')
        # print("path : ", path)
        yolo_pred, img = yolo_detector.detect(img)

        yolo_pred = yolo_pred[0]
        yolo_pred[:, :4] = scale_coords(img.shape[2:], yolo_pred[:, :4], final_img.shape).round()
        yolo_pred = yolo_pred.cpu().numpy()

        # yolo visualization
        for bbox in yolo_pred:
            cv2.rectangle(final_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(final_img, f'{yolo_class[int(bbox[-1])]}:{bbox[-2]:.2f}', (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            final_yolo.append([f'{path[-10:-4]}', yolo_class[int(bbox[-1])], torch.tensor([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]), bbox[4]])

        #center point
        print("---------------center point---------------")

        data = cp_detector.demo_dataset[idx]
        cp_pred = cp_detector.detect(data)
        cp_bboxes = cp_pred[0]['pred_boxes'].cpu().numpy()
        cp_labels = cp_pred[0]['pred_labels'].cpu().numpy()
        cp_score = cp_pred[0]['pred_scores'].cpu().numpy()
        cp_2d_boxes = []
        cp_cloud_point_boxes = []

        # set calibration matrix
        calib_file = f'/workspace/KITTI/calib/train/{path[-10:-4]}.txt'
        P2, R0, Tr = read_calib_file(calib_file)

        # read image
        image = cv2.imread(f'/workspace/KITTI/images/train/{path[-10:-4]}.png', cv2.IMREAD_ANYCOLOR)

        # cp projection and visualization
        for bbox, label, score in zip(cp_bboxes, cp_labels, cp_score):
            lidar_corners = lidar_box3d_to_corner3d(bbox)
            cp_box_x, cp_box_y = lidar3d_to_image2d_projection(lidar_corners, P2, R0, Tr)
            cp_box_2d = corners_to_2dbox(cp_box_x, cp_box_y)

            cp_x_min = max(cp_box_2d[0], 0)
            cp_y_min = max(cp_box_2d[1], 0)
            cp_x_max = min(cp_box_2d[2], final_img.shape[1] - 1)
            cp_y_max = min(cp_box_2d[3], final_img.shape[0] - 1)

            cp_2d_boxes.append([cp_x_min, cp_y_min, cp_x_max, cp_y_max])

            # cp box visualization
            cv2.rectangle(final_img, (cp_x_min, cp_y_min), (cp_x_max, cp_y_max), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(final_img, f'{cp_class[label]}:{score:.2f}', (cp_x_max, int((cp_y_min + cp_y_max)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(final_img, str(round(bbox[-1])), (int((cp_x_min + cp_x_max)/2), int((cp_y_min + cp_y_max)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) #angle

            final_cp.append([f'{path[-10:-4]}', cp_class[label], torch.tensor([cp_x_min, cp_y_min, cp_x_max, cp_y_max]), score])

            print("----------------------lidar projection------------------------")
            # lidar point cloud projection
            points_inside_box = []
            for point in data['points']:
                if all(lidar_corners.min(axis=0) <= point[:3]) and all(point[:3] <= lidar_corners.max(axis=0)):
                    points_inside_box.append(point[:3])

            points_inside_box = np.array(points_inside_box)

            XYZ1 = np.vstack([points_inside_box.T, np.ones((1, points_inside_box.shape[0]))])
            xyz = np.dot(P2,np.dot(R0,np.dot(Tr, XYZ1)))
            z = xyz[2, :]
            x = (xyz[0, :] / z).astype(np.int32)[z>0]
            y = (xyz[1, :] / z).astype(np.int32)[z>0]

            visualization_projection(final_img, x, y, (255, 100, 255,250))

            # 3d bbox corner projection
            visualization_3d_bbox(final_img, cp_box_x, cp_box_y, (200,0,200))


            # 3d bbox using projected lidar point cloud
            pj_x_min = max(min(x), 0)
            pj_y_min = max(min(y), 0)
            pj_x_max = min(max(x), final_img.shape[1] - 1)
            pj_y_max = min(max(y), final_img.shape[0] - 1)
            cp_cloud_point_boxes.append([pj_x_min, pj_y_min, pj_x_max, pj_y_max])
            cv2.rectangle(final_img, (pj_x_min, pj_y_min), (pj_x_max, pj_y_max), (200, 0, 200), thickness=2, lineType=cv2.LINE_AA)

        print("-----------------matching-----------------")

        indices_cp, indices_yolo = match_bboxes(torch.Tensor(cp_2d_boxes), torch.Tensor(yolo_pred[:, :4]))
        not_match_cp = [i for i in range(len(cp_2d_boxes)) if i not in indices_cp]
        not_match_yolo = [i for i in range(len(yolo_pred[:, :4])) if i not in indices_yolo]

        for idx_cp, idx_yolo in zip(indices_cp, indices_yolo):
            # label
            label_cp = cp_class[cp_labels[idx_cp]]
            label_yolo = yolo_class[int(yolo_pred[idx_yolo][-1])]

            # confidence
            conf_cp = cp_score[idx_cp]
            conf_yolo = yolo_pred[idx_yolo][4]

            if label_cp == label_yolo:
                label_match = label_cp
                conf_match = ((conf_cp * conf_cp) + (conf_yolo * conf_yolo)) / (conf_cp + conf_yolo)
            elif label_yolo in  ['Misc', 'DontCare'] and label_cp != 'Misc':
                label_match = label_cp
                conf_match = conf_cp
            else:
                label_match = label_yolo
                conf_match = conf_yolo

            # match bbox
            bbox_cp = cp_2d_boxes[idx_cp]
            bbox_yolo = yolo_pred[idx_yolo][ :4]

            new_box = weighted_box_fusion(bbox_cp, bbox_yolo, conf_cp, conf_yolo)

            final_pred.append([f'{path[-10:-4]}', label_match, torch.tensor(new_box), conf_match])

            if args.box_fusion:
                yolo_box.append([f'{path[-10:-4]}', label_match, torch.tensor(bbox_yolo), conf_match])
                cp_box.append([f'{path[-10:-4]}', label_match, torch.tensor(bbox_cp), conf_match])
                cp_cloud_point_box.append([f'{path[-10:-4]}',label_match, torch.tensor(cp_cloud_point_boxes[idx_cp]), conf_match])
                minmax_box.append([f'{path[-10:-4]}', label_match, torch.tensor(box_fusion_minmax(bbox_cp, bbox_yolo)), conf_match])
                conf_box.append([f'{path[-10:-4]}', label_match, torch.tensor(box_fusion_conf(bbox_cp, bbox_yolo, conf_cp, conf_yolo)), conf_match])
                weighted_box.append([f'{path[-10:-4]}', label_match, torch.tensor(weighted_box_fusion(bbox_cp, bbox_yolo, conf_cp, conf_yolo)), conf_match])

            # heading
            heading = calc_head(cp_bboxes[idx_cp][-1])

            # visualization
            tl = 1
            c1, c2 = (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3]))
            cv2.rectangle(image, c1, c2, colors[yolo_class.index(label_match)], thickness=tl, lineType=cv2.LINE_AA)

            tf = max(tl - 1, 1)  # font thickness
            text_match = f"{label_match} _match"
            text_yolo = f"{label_yolo} _yolo"
            text_cp = f"{label_cp} _cp"
            text_heading = f"{heading}"

            t_size = cv2.getTextSize(text_match, 0, fontScale=tl / 3, thickness=tf)[0]

            c2 = c1[0] + t_size[0], c1[1] - (t_size[1]*4) - 9
            cv2.rectangle(image, c1, c2, colors[yolo_class.index(label_match)], -1, cv2.LINE_AA)  # filled
            cv2.putText(image, text_heading, (c1[0], c1[1] - t_size[1]*3 - 8), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(image, text_match, (c1[0], c1[1] - t_size[1]*2 - 6), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(image, text_yolo, (c1[0], c1[1] - t_size[1] -4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(image, text_cp, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        for idx in not_match_cp:
            if cp_score[idx] > 0.8:
                label_match = cp_class[cp_labels[idx]]
                new_box = cp_2d_boxes[idx]
                heading = calc_head(cp_bboxes[idx][-1])

                final_pred.append([f'{path[-10:-4]}', label_match, torch.tensor(new_box), cp_score[idx]])

                # visualization
                tl = 1
                c1, c2 = (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3]))
                cv2.rectangle(image, c1, c2, colors[yolo_class.index(label_match)], thickness=tl, lineType=cv2.LINE_AA)

                tf = max(tl - 1, 1)  # font thickness
                text_match = f"{label_match} _match"
                text_cp = f"{label_match} _cp"
                text_heading = f"{heading}"

                t_size = cv2.getTextSize(text_match, 0, fontScale=tl / 3, thickness=tf)[0]

                c2 = c1[0] + t_size[0], c1[1] - (t_size[1]*4) - 9
                cv2.rectangle(image, c1, c2, colors[yolo_class.index(label_match)], -1, cv2.LINE_AA)  # filled
                cv2.putText(image, text_heading, (c1[0], c1[1] - t_size[1]*3 - 8), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(image, text_match, (c1[0], c1[1] - t_size[1]*2 - 6), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(image, text_cp, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        for idx in not_match_yolo:
            if yolo_pred[idx][4] > 0.6:
                label_match = yolo_class[int(yolo_pred[idx][-1])]
                new_box = yolo_pred[idx][ :4]
                final_pred.append([f'{path[-10:-4]}', label_match, torch.tensor(new_box), yolo_pred[idx][4]])

                # visualization
                tl = 1
                c1, c2 = (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3]))
                cv2.rectangle(image, c1, c2, colors[yolo_class.index(label_match)], thickness=tl, lineType=cv2.LINE_AA)

                tf = max(tl - 1, 1)  # font thickness
                text_match = f"{label_match} _match"
                text_yolo = f"{label_match} _yolo"

                t_size = cv2.getTextSize(text_match, 0, fontScale=tl / 3, thickness=tf)[0]

                c2 = c1[0] + t_size[0], c1[1] - (t_size[1]*4) - 9
                cv2.rectangle(image, c1, c2, colors[yolo_class.index(label_match)], -1, cv2.LINE_AA)  # filled
                cv2.putText(image, text_heading, (c1[0], c1[1] - t_size[1]*3 - 8), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(image, text_match, (c1[0], c1[1] - t_size[1]*2 - 6), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(image, text_yolo, (c1[0], c1[1] - t_size[1] -4), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


        print("---------------evaluation-----------------")

        # get label data
        # kitti format : [class_type, truncated, occluded, alpha, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, dimension_height, dimension_width, dimension_length, location_x, location_y, location_z, rotation]

        label_file = f'/workspace/KITTI/kitti_labels/train/{path[-10:-4]}.txt'
        with open(label_file, 'r') as f:
            lines = [x.split() for x in f.read().strip().splitlines()]
            for line in lines:
                label_class = line[0]
                gt_pred.append([f'{path[-10:-4]}', label_class, torch.tensor([np.float32(line[4]), np.float32(line[5]), np.float32(line[6]), np.float32(line[7])])]) # file name, label_class, (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)

                # gt visualization
                cv2.rectangle(final_img, (int(float(line[4])), int(float(line[5]))), (int(float(line[6])), int(float(line[7]))), (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(final_img, f'{label_class}', (int(float(line[4])), int(float(line[7]))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imwrite(f"./nahye_output/{path[-10:-4]}.png", final_img)
        cv2.imwrite(f"./nahye_output/{path[-10:-4]}_matching.png", image)
        print(f"saved image : {path[-10:-4]}")

    print("==========================yolo==========================")
    make_confusion_matrix(file_list, final_yolo, gt_pred)
    print("==========================center point==========================")
    make_confusion_matrix(file_list, final_cp, gt_pred)
    print("==========================fusion==========================")
    make_confusion_matrix(file_list, final_pred, gt_pred)

    if args.box_fusion:
        print("==========================yolo_iou==========================")
        yolo_iou = evaluate_box_fusion(file_list, yolo_box, gt_pred)
        print("==========================cp_iou==========================")
        cp_iou = evaluate_box_fusion(file_list, cp_box, gt_pred)
        print("==========================cp_cloud_point_iou==========================")
        cp_cloud_point_iou = evaluate_box_fusion(file_list, cp_cloud_point_box, gt_pred)
        print("==========================minmax_iou==========================")
        minmax_iou = evaluate_box_fusion(file_list, minmax_box, gt_pred)
        print("==========================conf_iou==========================")
        conf_iou = evaluate_box_fusion(file_list, conf_box, gt_pred)
        print("==========================weighted_iou==========================")
        weighted_iou = evaluate_box_fusion(file_list, weighted_box, gt_pred)

        print("==========================total==========================")
        print("yolo_iou : ", yolo_iou)
        print("cp_iou : ", cp_iou)
        print("cp_cloud_point_iou : ", cp_cloud_point_iou)
        print("minmax_iou : ", minmax_iou)
        print("conf_iou : ", conf_iou)
        print("weighted_iou : ", weighted_iou)

    print("finish!!!!")
