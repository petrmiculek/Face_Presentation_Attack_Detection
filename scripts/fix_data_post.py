"""
Exploring generated face crops, detecting faulty ones.
Re-running of MTCNN face detection
Image orientation check
Laplacian variance check
Landmark distance check
"""
# stdlib
import argparse
import os
import sys
from os.path import join
from shutil import move

# fix for local import problems
cwd = os.getcwd()
sys.path.extend([cwd] + [join(cwd, d) for d in os.listdir() if os.path.isdir(d)])

# external
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2

# local
from util_torch import init_device
from face_detection import get_ref_landmarks, face_height, find_rotation, rotate
from util_image import plot_many

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', help='directory with images', type=str,
                    required=True)  # e.g. '/mnt/sdb1/dp/rose_youtu_vids'


def plot_landmarks(img, landmarks=None, title=None):
    """ Show image with the detected landmarks. """
    plt.imshow(img)
    if landmarks is not None:
        if not isinstance(landmarks, list):
            landmarks = [landmarks]
        for l in landmarks:
            if l is not None:
                plt.scatter(l[:, 0], l[:, 1])
    if title:
        plt.title(title)
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':
    ''' Parse arguments '''
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))
    # human-readable numpy float printing
    np.set_printoptions(precision=3, suppress=True)
    ''' Video and Image paths '''
    dir_imgs = args.input_path
    df = pd.read_pickle(join(dir_imgs, 'data.pkl'))
    os.makedirs(join(dir_imgs, 'fixed'))
    os.makedirs(join(dir_imgs, 'removed'))

    ''' Face Alignment parameters '''
    scale = 3.25
    margin = 384 - 112 * scale
    ref_pts = get_ref_landmarks(scale, margin)
    ref_height = face_height(ref_pts)
    img_size = (384, 384)
    ''' MTCNN '''
    device = init_device()
    mtcnn = MTCNN(image_size=384, select_largest=True, device=device, post_process=False, min_face_size=150)

    ''' Intermediate lists '''
    to_remove = []  # keep indexes, paths and info about faulty images
    # manuals = []
    # keep = []

    ''' Loop through annotations '''
    try:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            img = np.array(Image.open(join(dir_imgs, row['path'])))
            bbox_orig = row['box']
            info = ''
            flagged = False
            landmarks_sample = None
            ''' Estimate blur '''
            blur_val = cv2.Laplacian(img, cv2.CV_64F).var()
            # if blur_val < 20:
            #     flagged = True
            #     info += f' blur{blur_val:.0f}'

            ''' Check face orientation '''
            rot = find_rotation(mtcnn, img)
            if rot is None:
                flagged = True
                info += f' no_rot'
            else:
                if rot != 0:
                    flagged = True
                    info += f' rotated_{rot:.0f}'

                ''' Fix face orientation/rotation '''
                img = rotate(img, rot)

                # re-run MTCNN prediction
                b, probs, lms = mtcnn.detect(img, landmarks=True)
                # ^ only one prediction (at most), because of the select_largest flag
                # single-image batch
                if b is None:
                    info += f' no_detection'
                else:
                    # check landmarks
                    b = b[0]
                    lms = lms[0].astype(np.float32)
                    landmarks_sample = lms

                    # landmarks height
                    h = face_height(lms)
                    height_wrong = h < 0.8 * ref_height or h > 1.2 * ref_height
                    # distance between predicted and reference landmarks
                    dist_lms = np.mean(np.linalg.norm(lms - ref_pts, axis=1))
                    dist_wrong = dist_lms > 15
                    metadata_dict = {
                        'i': i,
                        'path': row['path'],
                        'box': b,
                        'landmarks': lms,
                        'probs': probs[0],
                        'height': h,
                        'dist': dist_lms,
                        'blur': blur_val}
                    if dist_wrong:
                        flagged = True
                        info += f' landmarks{dist_lms:.0f}'

                    if height_wrong:
                        flagged = True
                        info += f' height{h:.0f}'

                    if not height_wrong and not dist_wrong:
                        # keep.append(metadata_dict)
                        ''' Save fixed rotation '''
                        if rot != 0:
                            cv2.imwrite(join(dir_imgs, 'fixed', row['path']), img)
                            flagged = False  # don't remove if everything else was OK.

            # end of mega-nested ifs
            if flagged:
                to_remove.append({'i': i, 'path': row['path'], 'info': info})
                if False:
                    # plt.imshow(img)
                    # if landmarks_sample is not None:
                    #     plt.scatter(landmarks_sample[:, 0], landmarks_sample[:, 1], c='g')
                    # plt.scatter(ref_pts[:, 0], ref_pts[:, 1], c='r')
                    # plt.title(info)
                    # plt.tight_layout(pad=0)
                    # plt.show()
                    #
                    # s = input()
                    # if len(s) > 0:
                    #     manuals.append({'i': i, 'path': row['path'], 'note': s})
                    pass
    except:
        pass

    ''' Re-save the good ones '''
    to_remove = pd.DataFrame(to_remove)
    df = df.drop(np.array(to_remove.i.values))
    df = df.reset_index(drop=True)

    df.to_pickle(join(dir_imgs, 'data_filtered.pkl'))
    path_to_remove = join(dir_imgs, 'to_remove.pkl')
    if not os.path.exists(path_to_remove):
        to_remove.to_pickle(path_to_remove)

    # remove the bad ones
    for i, row in tqdm(to_remove.iterrows(), total=len(to_remove)):
        path_to_remove = join(dir_imgs, row['path'])
        # os.remove(path_to_remove)
        move(path_to_remove, join(dir_imgs, 'removed', row['path']))
