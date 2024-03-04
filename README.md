# Camera & 3D LiDAR Object Detection
> **camera와 3d LiDAR sensor fusion을 통한 object detection**
>
> RGB image와 3D point cloud 데이터를 사용하여 객체 인식 모델을 학습하고 data association 수행
>
> 개발 기간 : 2024.01 ~ 2024. 02

## Table of Contents
1. 프로젝트 소개
2. 실행 방법
3. 결과
4. 전체 과정
5. 파일 구조
6. 결과 영상
7. Stacks


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
     ├── labels <-- class, 2D bounding box, dimension, location, rotation 정보
     |	└── training
     ├── velodyne <-- LiDAR point cloud 데이터
     |	├── testing
     |	└── training
     ├── train_cp.txt <-- centerpoint 모델을 위한 train 데이터 경로 리스트
     ├── train_yolo.txt <-- yolo 모델을 위한 train 데이터 경로 리스트
     ├── val_cp.txt <-- centerpoint 모델을 위한 val 데이터 경로 리스트
     ├── val_yolo.txt <-- yolo 모델을 위한 val 데이터 경로 리스트
     └── test.txt
```

#### class
- 9개 class
- 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', ‘DontCare’

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

### 4-2. YOLO v7
- [YOLO v7 모델](https://github.com/WongKinYiu/yolov7)
- 2D 이미지에서 객체를 인식하기 위해 yolo v7 모델을 사용한다
- dataset의 경우, KITTI format을 YOLO format으로 변환하여 사용한다 (KITTI/kitti2yolo.py 파일)

![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/f454c9f0-f380-4055-9f7b-3677931cd75e)

### 4-3. CenterPoint
- [CenterPoint 모델](https://github.com/tianweiy/CenterPoint-KITTI)
- LiDAR의 3D 데이터 즉, point cloud 에서 객체를 인식하기 위해 centerpoint 모델을 사용한다

![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/6736ccf5-64c3-4c37-9b07-2a3cc5637dd0)

### 4-4. 3D Bounding Box Projection
- CenterPoint 모델에서 예측한 3D bounding box를 2D image에 projection 한다

![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/836c6661-d407-4ca4-ac39-094596bd98f1)

#### Projection 과정
![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/f06a09e8-1716-4bc8-ba63-9072d189dc25)

1. 3D bounding box 위치 (x, y, z), dimension (width, height, length), yaw 값을 통해 3D bounding box의 8개 꼭지점 추출
2. 3D 상의 8개 꼭지점을 2D image 영역으로 projection
3. min, max를 이용하여 4개의 꼭지점을 가진 2D bounding box 추출

### 4-4. Object matching
- CenterPoint에서 예측한 object와 YOLO에서 예측한 object를 융합한다
- IoU 기반 헝가리안 알고리즘을 활용해 두 모델의 검출 결과에서 같은 객체를 추출한다
- 한 모델만 객체를 검출한 경우 confidence 값에 따라 객체로 추출한다

#### Bounding box matching
- Weighted Box Fusion 방법을 활용해 두 모델의 bounding box를 하나로 합친다
- $new\\_box = \frac{box1\\_points * box1\\_conf + box2\\_points * box2\\_conf}{box1\\_conf + box2\\_conf}$
<br/>

- box를 하나로 합치는 방법을 정하기 위해 아래와 같이 여러 방법을 수행하고 그때의 CIoU 값을 비교하였다
- 그 결과 WBF 방법이 가장 좋았기 때문에 이 방법을 선택하였다
![image](https://github.com/nahye03/Camera_3DLiDAR_Object_Detection/assets/54797864/c41e0f97-804b-411f-a882-1949e12447d4)

#### Class
- YOLO 모델의 class를 기본으로 사용한다
- 단, yolo 모델의 결과가 'Dontcare', 'Misc'이고, centerpoint 모델의 결과가 'Misc'가 아닐 때는 centerpoint 모델의 class 사용

#### Confidence
- weighted sum 방식으로 yolo의 confidence와 centerpoint의 confidence를 합친다
- 한 모델에서만 object를 검출했을 경우에는 그때의 confidence를 그대로 사용한다

## 5. 파일 구조
```
└─ KITTI <-- dataset
└─ yolo_cp
     ├── models <-- yolo model
     ├── pcdet <-- centerpoint 모델 관련 라이브러리
     ├── result_images <-- 결과 이미지들
     ├── cpconv <-- centerpoint 모델 관련 라이브러리
     ├── tools/cfg <-- centerpoint model
     ├── utils <-- yolo 모델 관련 라이브러리
     ├── best.pt <-- yolo 모델 weight
     ├── checkpoint_epoch_80_yolodata.pth <-- centerpoint 모델 weight
     ├── lidar_projection.py <-- lidar point cloud를 이미지로 projection, 시각화
     └── our_demo_nahye.py <-- main 파일
```


## 6. 결과 영상
[![Video Label](http://img.youtube.com/vi/p0FAgaJ8ODk/0.jpg)](https://youtu.be/p0FAgaJ8ODk)


## 7. Stacks
### Environment
<img src="https://img.shields.io/badge/ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"> <img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/amazonec2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"> 

### Config
<img src="https://img.shields.io/badge/yaml-CB171E?style=for-the-badge&logo=yaml&logoColor=white">

### Development
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
