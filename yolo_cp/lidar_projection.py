import cv2
import numpy as np
import open3d
import argparse
import matplotlib
import matplotlib.pyplot 
import matplotlib.pyplot as plt
from mayavi import mlab



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', required = True)
    args = parser.parse_args()

    return args

def read_calib_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    P2 = np.array([float(i) for i in lines[2].split(' ')[1:]]).reshape(3,4)
    R0_rect = np.array([float(i) for i in lines[4].split(' ')[1:]]).reshape(3,3)
    Tr_velo_to_cam = np.array([float(i) for i in lines[5].split(' ')[1:]]).reshape(3,4)

    return P2, R0_rect, Tr_velo_to_cam

def visualization_open3d(data):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data[:, :3])
    open3d.visualization.draw_geometries([pcd])

def visualization_plt(image_file, data, x, y):
    img = cv2.imread(image_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aspect_ratio = float(img.shape[1]) / img.shape[0]

    fig, axs = matplotlib.pyplot.subplots(1, 2, figsize=(20, 25 ))

    axs[0].imshow(img_rgb)
    axs[0].axis('off')

    x_values = data[:, 0]

    x_min, x_max = np.percentile(x_values, 1), np.percentile(x_values, 99)

    scatter = axs[1].scatter(x, img.shape[0] - y, c=x_values, cmap='jet', marker='.', s=15, vmin=x_min, vmax=x_max)
    axs[1].set_xlim([0, img.shape[1]])
    axs[1].set_ylim([0, img.shape[0]])
    axs[1].axis('off')

    for ax in axs:
        ax.set_aspect(aspect_ratio)

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

def visualization_projection(image_file, data, x, y):
    img = cv2.imread(image_file)
    img_mapped = img.copy()
    img_h, img_w = img.shape[:2]

    x_normalized = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
    colors = plt.cm.magma(x_normalized)

    for i, (ix, iy) in enumerate(zip(x, y)):
        if 0 <= ix < img_w and 0 <= iy < img_h:
            color = (colors[i] * 255).astype(np.uint8)[:3]
            color = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(img_mapped, (ix, iy), radius=1, color=color, thickness=2)

    img_mapped_rgb = cv2.cvtColor(img_mapped, cv2.COLOR_BGR2RGB)

    plt.imshow(img_mapped_rgb)
    plt.show()

def visualization_mayavi(data):
    # filtered_data = data[data[:, 0] >= 0]
    filtered_data = data

    x = filtered_data[:, 0]
    y = filtered_data[:, 1]
    z = filtered_data[:, 2]

    mlab.figure(bgcolor=(0, 0, 0))
    mlab.points3d(x, y, z, color=(0, 1, 0), mode='point')
    # mlab.axes()
    mlab.show()

if __name__ == "__main__":
    args = get_args()

    file_name = args.file_name
    calib_file = f'/workspace/KITTI/calib/train/{file_name}.txt'
    image_file = f'/workspace/KITTI/images/train/{file_name}.png'
    velo_file = f'/workspace/KITTI/velodyne/train/{file_name}.bin'

    P2, R0_rect, Tr_velo_to_cam = read_calib_file(calib_file)
    R0 = np.eye(4)
    R0[:3, :3] = R0_rect
    Tr = np.vstack([Tr_velo_to_cam, [0,0,0,1]])

    with open(velo_file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32).reshape(-1,4)

    XYZ1 = np.vstack([data[:,:3].T, np.ones((1, data.shape[0]))])
    xyz = np.dot(P2,np.dot(R0,np.dot(Tr, XYZ1)))
    z = xyz[2, :]
    x = (xyz[0, :] / z).astype(np.int32)[z>0]
    y = (xyz[1, :] / z).astype(np.int32)[z>0]

    print(xyz)

    visualization_open3d(data)
    visualization_plt(image_file, data[z>0], x, y)
    visualization_projection(image_file, data[z>0], x, y)
    visualization_mayavi(data)
