"""
Exploring generated face crops, detecting faulty ones.
Re-running of MTCNN face detection
Image orientation check
Laplacian variance check
"""
# stdlib
import os
import sys
from os.path import join

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

# local
from util_torch import init_device
from util_face import get_ref_landmarks, face_height, get_align_transform, transform_bbox, find_rotation, rotate
from util import plot_many

if __name__ == '__main__':
    # human-readable numpy float printing
    np.set_printoptions(precision=3, suppress=True)
    ''' MTCNN '''
    # MTCNN for re-doing the predictions, if I want to:
    device = init_device()
    mtcnn = MTCNN(image_size=384, select_largest=True, device=device, post_process=False, min_face_size=50)
    dir_imgs = '/mnt/sdb1/dp/rose_youtu_imgs'
    df = pd.read_pickle(join(dir_imgs, 'data.pkl'))
    # dir_out = '/mnt/sdb1/dp/rose_youtu_single_aligned'
    # os.makedirs(dir_out, exist_ok=False)
    scale = 3.25
    margin = 384 - 112 * scale
    ref_pts = get_ref_landmarks(scale, margin)
    ref_height = face_height(ref_pts)
    img_size = (384, 384)
    no_bbox = []
    landmarks_far = []
    info_all = []
    no_rots = []
    blurs = []
    df = df[::100]
    i, row = 0, df.iloc[0]
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img = np.array(Image.open(join(dir_imgs, row['path'])))
        bbox_orig = row['box']
        blur_val = cv2.Laplacian(img, cv2.CV_64F).var()
        blurs.append({'path': row['path'], 'blur': blur_val})

        rot = find_rotation(mtcnn, img)
        if rot is None:
            no_rots.append({'i': i, 'path': row['path'], 'blur': blur_val})
            continue

        img = rotate(img, rot)

        # measure blurriness
        if blur_val < 20:
            plot_many(img, title=f'blur: {blur_val:.0f}')
            _ = input()

        # re-run MTCNN prediction
        batch = img
        b, probs, lms = mtcnn.detect(batch, landmarks=True)
        # ^ only one prediction (at most), because of the select_largest flag
        # single-image batch
        if b is None:
            no_bbox.append({'i': i, 'path': row['path'],  # , 'box': bbox, 'landmarks': landmarks
                            'bbox_orig': bbox_orig})
            continue
        # check landmarks
        b = b[0]
        lms = lms[0].astype(np.float32)

        # landmarks height
        h = face_height(lms)
        height_wrong = h < 0.8 * ref_height or h > 1.2 * ref_height
        # distance between predicted and reference landmarks
        dist = np.linalg.norm(lms - ref_pts, axis=1)
        dist_wrong = np.mean(dist) > 15
        info = {'i': i,
                'path': row['path'],
                'box': b,
                'landmarks': lms,
                'probs': probs[0],
                'height': h,
                'dist': dist,
                'blur': blur_val
                }
        if any([height_wrong, dist_wrong]):
            landmarks_far.append(info)

        info_all.append(info)

        if False:
            plt.imshow(img)
            plt.scatter(l[:, 0], l[:, 1], c='r')
            plt.scatter(ref_pts[:, 0], ref_pts[:, 1], c='b')
            plt.show()
            plt.close()

        # break
    landmarks_far = pd.DataFrame(landmarks_far)

    plt.hist(landmarks_far['height'], bins=100)
    plt.title('landmarks height')
    # vertical line at ref_height
    plt.axvline(ref_height, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    no_bbox = pd.DataFrame(no_bbox)
    iters = landmarks_far.iterrows()
    iters = no_bbox.iterrows()

    i, r = next(iters)
    img = Image.open(join(dir_imgs, r['path']))
    plt.imshow(np.array(img))
    rot_angle = find_rotation(mtcnn, np.array(img))
    plt.imshow(img)
    lms = r['landmarks']
    plt.scatter(lms[:, 0], lms[:, 1], c='r')
    plt.show()

    no_rots = pd.DataFrame(no_rots)

    imgs = [Image.open(join(dir_imgs, r['path'])) for i, r in no_rots.iterrows()]

    plot_many(imgs, titles=[f'{b:.2f}' for b in no_rots.blur])

    blurs = pd.DataFrame(blurs)
    bc = blurs.copy()
    blurs.sort_values('blur', inplace=True)

    n = 0
    blurs_head = blurs[n:n + 12]
    imgs = [Image.open(join(dir_imgs, r['path'])) for i, r in blurs_head.iterrows()]
    plot_many(imgs, titles=[f'{b:.2f}' for b in blurs_head.blur])
    n += 12

    # show 9 samples of low laplacian variance (blurry ones)
    blurs_samples = blurs[:len(blurs) // 4: len(blurs) // 34]
    imgs = [np.array(Image.open(join(dir_imgs, r['path']))) for i, r in blurs_samples.iterrows()]
    imgs_rotated = []
    rots = [find_rotation(mtcnn, img) for img in imgs]
    for img, ro in zip(imgs, rots):
        if ro is not None:
            imgs_rotated.append(rotate(img, ro))
        else:
            imgs_rotated.append(img)

    import seaborn as sns

    sns.set_context('talk')
    plot_many(imgs_rotated, titles=[f'{b:.2f}' for b in blurs_samples.blur],
              output_path='laplacian_blurry.pdf')
