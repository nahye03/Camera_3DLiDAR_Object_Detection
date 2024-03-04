# Camera & 3D LiDAR Object Detection
> **camera와 3d LiDAR sensor fusion을 통한 object detection**
>
> RGB image와 3D point cloud 데이터를 사용하여 객체 인식 모델을 학습하고 data association 수행
>
> 개발 기간 : 2024.01 ~ 2024. 02

## Table of Contents

## 1. 프로젝트 소개
![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/894b8c6d-aa95-44fc-9280-d9087abbefea)

- 2D 이미지를 기반으로 YOLO v7 모델을 학습하며 2D Object Detection을 수행한다
- 3D Point Cloud를 기반으로 centerpoint 모델을 학습하여 3D Object Detection을 수행한다
- 3D Object와 2D Object 간에 data association을 통해 object detection 성능을 높인다
- LiDAR 정보를 사용하여 object의 depth를 구한다

## 2. 실행 방법
`python our_demo_nahye.py --cfg_file [centerpoint 모델] --ckpt [centerpoint weight] --data_path [centerpoint dataset] --weight [YOLO v7 weight] --conf [confidence 값] --img-size [YOLO v7 input image size] --source [YOLO v7 dataset]`

```
python our_demo_nahye.py --cfg_file /workspace/yolo_cp/tools/cfgs/kitti_models/centerpoint_yolo.yaml --ckpt checkpoint_epoch_80_yolodata.pth --data_path /workspace/KITTI/velodyne/train --weight best.pt --conf 0.25 --img-size 640 --source /workspace/KITTI/images/train
```
- 여러 box fusion 방법에 대해 실행하고 싶으면 `--box_fusion` 옵션 추가

## 3. 결과
object class, bounding box, depth에 대해 추정한다

![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/55cd3239-3d32-46f7-b330-ec2ca32c1793)

### object detection 결과 mAP
- 단일 센서를 사용하는 방법보다 카메라와 LiDAR를 융합한 방법이 더 좋다

![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/89000a1a-4cbc-4041-a476-c6d3ed6668df)

### Depth
객체와의 depth 정확도는 99.21%

### Heading
- Object가 향하는 방향인 heading 정확도는 91.67%
- 이때 heading 방향은 아래와 같이 정의하였다
  
![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/e1b7ddf4-1a7c-410c-a0ff-f6eeacac5d26)

## 4. 전체 과정
### 4-1. Dataset
- KITTI의 3D Object 데이터셋을 다운받아서 사용했다

#### dataset 폴더 구조
```
└─ KITTI
     ├── calib <-- camera와 LiDAR 간의 calibration (camera calibration matrices of object data set)
     |	├── testing
     |	└── training
     ├── images <-- RGB 이미지 데이터
     |	├── testing
     |	└── training
     ├── labels <-- class, bounding box 관한 정보
     |	└── training
     └── velodyne <-- LiDAR point cloud 데이터
      	├── testing
       	└── training
```

#### calib 데이터 구조
- 텍스트 파일
- P0, P1, P2, P3
  - projection 행렬 (world to image)
  - 12개의 값 -> 3x4 행렬
  - 본 프로젝트에서는 cam2 이미지 데이터를 사용하기 때문에 `P2`만 사용한다
- R0_rect
  - world 평면으로 회전시켜주는 회전 변환 행렬
  - 9개의 값 -> 3x3 행렬
- Tr_velo_to_cam
  - LiDAR to camera 변환 행렬
  - 9개의 값 -> 3x3 행렬

## 5. 파일 구조


## 6. Stacks
### Environment
<img src="https://img.shields.io/badge/ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"> <img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/amazonec2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"> 

### Config
<img src="https://img.shields.io/badge/yaml-CB171E?style=for-the-badge&logo=yaml&logoColor=white">

### Development
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
