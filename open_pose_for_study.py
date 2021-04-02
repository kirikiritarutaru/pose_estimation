from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from utils.decode_pose import decode_pose
from utils.openpose_net import OpenPoseNet

size = (368, 368)
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]


def preprocess(org_img: np.ndarray) -> np.ndarray:
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    p_img = cv2.resize(org_img, size, interpolation=cv2.INTER_CUBIC)

    p_img = p_img.astype(np.float32)/255.

    pre_img = p_img.copy()

    # 絶対ベクトル化できるだろ
    for i in range(3):
        pre_img[:, :, i] = pre_img[:, :, i]-color_mean[i]
        pre_img[:, :, i] = pre_img[:, :, i]/color_std[i]

    p_img = pre_img.transpose((2, 0, 1)).astype(np.float32)

    return p_img


if __name__ == '__main__':
    gpu_on = torch.cuda.is_available()
    if gpu_on:
        device = 'cuda:0'
    else:
        device = 'cpu'

    net = OpenPoseNet()
    net = net.to(device)

    net_weights = torch.load(
        'pose_model_scratch.pth',
        map_location={'cuda:0': 'cpu'}
    )
    keys = list(net_weights.keys())

    weights_load = {}

    for i in range(len(keys)):
        weights_load[list(net.state_dict().keys())[i]] = (
            net_weights[list(keys)[i]]
        )

    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        t1 = time()
        ret, img = cap.read()
        prepr_img = preprocess(img)
        prepr_img = torch.from_numpy(prepr_img)
        x = prepr_img.unsqueeze(0)
        if gpu_on:
            x = x.to(device)

        with torch.no_grad():
            net.eval()
            predicted_outputs, _ = net(x)

        if gpu_on:
            pafs = predicted_outputs[0][0].to(
                'cpu').detach().numpy().transpose(1, 2, 0)
            heatmaps = predicted_outputs[1][0].to(
                'cpu').detach().numpy().transpose(1, 2, 0)
        else:
            pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
            heatmaps = predicted_outputs[1][0].detach(
            ).numpy().transpose(1, 2, 0)

        pafs = cv2.resize(
            pafs,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        heatmaps = cv2.resize(
            heatmaps,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        _, result_img, _, _ = decode_pose(img, heatmaps, pafs)
        cv2.imshow('output', result_img)

        t2 = time()
        print(f'processing time: {t2-t1:.3f} sec')
        k = cv2.waitKey(1)

        if k == ord('q') or not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
