# given a video, boxes and a tracking function, plot tracking
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import subprocess
import glob
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
from sort import Sort
import xml.etree.ElementTree as ET
import motmetrics as mm
from time import time
from copy import deepcopy
from podm.podm import get_pascal_voc_metrics, BoundingBox, MetricPerClass
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import fclusterdata
from sklearn.metrics import homogeneity_score


def annotationXmlToDict(xml_filename):
    etree = ET.parse(xml_filename)
    root = etree.getroot()
    result = {'pollen': []}
    result['filename'] = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    result['width'] = width
    result['height'] = height
    objects = root.findall('object')
    for object in objects:
        box = object.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        box_dict = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        class_name = object.find('name').text
        if class_name == 'pollen':
            result['pollen'].append(box_dict)
        elif class_name == 'gray':
            result['gray'] = box_dict
        elif class_name == 'gray1':
            result['gray1'] = box_dict
        else:
            print("WHAT ? ", class_name)
    return result


def cropPollenImages(save=True, colorcorrect=False):
    target_path = os.path.join('out_pollen_colorcorrection', 'corrected' if colorcorrect else 'original')
    path = 'data/colorcorrection_labels'
    imgs = []
    for filename in os.listdir(path):
        if not filename.endswith('.xml'):
            continue
        if filename in ['Erlen_Hive_04_rl_c2pro_Rec_20201122_124104_410_M.mp4_1500.xml',
                        'Erlen_Hive_04_rl_c2pro_Rec_20201122_124104_410_M.mp4_4125.xml',
                        'Erlen_Hive_04_rl_c2pro_Rec_20201122_125607_410_M.mp4_7150.xml',
                        'Erlen_Hive_04_rl_c2pro_Rec_20201122_130556_410_M.mp4_75.xml']:
            continue
        imgs.append([])
        result = annotationXmlToDict(os.path.join(path, filename))
        img_filename = result['filename']
        img = cv2.imread(os.path.join('data/colorcorrection_pollen_imgs', img_filename))
        if colorcorrect:
            gray_box = result['gray1']
            reference_img = img[gray_box['ymin']:gray_box['ymax'], gray_box['xmin']:gray_box['xmax']]
            gray = np.mean(np.reshape(reference_img.astype(float), (-1, 3)), axis=0)
            lum = np.mean(gray)
            # lum = 100.0
            # lum = 153.0
            for c in range(3):
                # print(lum, gray[c])
                img[:, :, c] = (img[:, :, c] * lum / gray[c]).clip(0, 255)
        for i, pollen in enumerate(result['pollen']):
            crop_img = img[pollen['ymin']:pollen['ymax'], pollen['xmin']:pollen['xmax']]
            imgs[-1].append(crop_img)
            if save:
                cv2.imwrite(os.path.join(target_path, f'{img_filename}{i}.png'), crop_img)
    imgs = [[img_list[i] for img_list in imgs] for i in range(9)]
    return imgs


def plotPollenGrid(pollen_imgs, colorcorrectpath='original'):
    out_path = f'out_pollen_colorcorrection/{colorcorrectpath}_grid'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for i, imgs in enumerate(pollen_imgs):
        cols = 5
        f, axarr = plt.subplots(max(int(len(imgs) * 1.0 / cols + 0.99), 2), cols)
        for j, img in enumerate(imgs):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axarr[j // cols, j % cols].imshow(img)
        plt.savefig(f'{out_path}/{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def plotPollenColorHistogramsSingle(pollen_imgs, colorcorrectpath='original'):
    out_path = f'out_pollen_colorcorrection/{colorcorrectpath}_hist0'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # color histograms
    for i, imgs in enumerate(pollen_imgs):
        plt.figure()
        plt.xlim([0, 256])
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color = ('r', 'g', 'b')
            for c, col in enumerate(color):
                histr = cv2.calcHist([img], [c], None, [256], [0, 256])
                histr = histr / img.size * img.shape[-1]
                plt.plot(histr, color=col)
        plt.savefig(f'{out_path}/{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def plotPollenColorHistogramsCombined(pollen_imgs, colorcorrectpath='original'):
    out_path = f'out_pollen_colorcorrection/{colorcorrectpath}_hist1'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # color histograms
    for i, imgs in enumerate(pollen_imgs):
        plt.figure()
        plt.xlim([0, 256])
        total_hist = [0, 0, 0]
        color = ('b', 'g', 'r')
        for img in imgs:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for c, col in enumerate(color):
                histr = cv2.calcHist([img], [c], None, [256], [0, 256])
                histr = histr / img.size * img.shape[-1]
                total_hist[c] = histr + total_hist[c]
                # plt.plot(histr, color=col)
        for c, col in enumerate(color):
            plt.plot(total_hist[c], color=col)
        plt.savefig(f'{out_path}/{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def plotPollenColorGrid(pollen_imgs, colorcorrect=False, custompath=None):
    colorcorrectpath = 'corrected' if colorcorrect else 'original'
    if custompath is not None:
        colorcorrectpath = custompath

    plotPollenGrid(pollen_imgs, colorcorrectpath=colorcorrectpath)
    plotPollenColorHistogramsSingle(pollen_imgs, colorcorrectpath=colorcorrectpath)
    plotPollenColorHistogramsCombined(pollen_imgs, colorcorrectpath=colorcorrectpath)


def clusterColors(pollen_imgs):
    color = ('b', 'g', 'r')
    X = []
    y = []
    classes = [0, 0, 1, 2, 3, 3, 4, 5, 0]
    for j, imgs in enumerate(pollen_imgs):
        for img in imgs:
            y.append(classes[j])
            img_center = img[int(img.shape[0] * 0.2):int(img.shape[0] * 0.8),
                         int(img.shape[1] * 0.2):int(img.shape[1] * 0.8)]
            img_center = img
            hists = np.array(
                [cv2.calcHist([img_center], [c], None, [256], [0, 256]) for c, col in enumerate(color)]).reshape(
                -1)
            hists = hists * 1.0 / img_center.size * img_center.shape[-1]
            # X.append(hists)

            margin = 0.2
            img_center = img[int(img.shape[0] * margin):int(img.shape[0] * (1 - margin)),
                         int(img.shape[1] * margin):int(img.shape[1] * (1 - margin))]
            avg_col = img_center.mean(axis=(0, 1))
            X.append(avg_col)

    X = np.array(X)

    def colorhistr_dist(p1, p2):
        # return ((p1-p2)**2).mean()
        split_idx = [0, 256 * 1, 256 * 2, 256 * 3]
        dists = [wasserstein_distance(p1[split_idx[i]:split_idx[i + 1]], p2[split_idx[i]:split_idx[i + 1]]) for i in
                 range(len(split_idx) - 1)]
        dist = np.mean(dists)
        return dist

    def color_dist(p1, p2):
        return ((p1 - p2) ** 2).mean()

    fclust1 = fclusterdata(X, t=6, criterion='maxclust', metric=color_dist, method='average')
    # print(fclust1)
    print("Homogeneity score:", homogeneity_score(y, fclust1))
    return fclust1


if __name__ == "__main__":
    if not os.path.exists('out_pollen_colorcorrection'):
        os.mkdir('out_pollen_colorcorrection')

    colorcorrect = True
    pollen_imgs = cropPollenImages(save=True, colorcorrect=colorcorrect)
    plotPollenColorGrid(pollen_imgs, colorcorrect=colorcorrect)

    clusters = clusterColors(pollen_imgs)

    clustered_imgs = [[] for _ in range(max(clusters))]
    for c, img in zip(clusters, [x for y in pollen_imgs for x in y]):
        clustered_imgs[c - 1].append(img)

    plotPollenColorGrid(clustered_imgs, custompath='cluster_' + ('corrected' if colorcorrect else 'original'))
