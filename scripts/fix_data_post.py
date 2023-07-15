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
import cv2

# local
from util_torch import init_device
from util_face import get_ref_landmarks, face_height, get_align_transform, transform_bbox, find_rotation, rotate
from util_image import plot_many

if __name__ == '__main__':
    # human-readable numpy float printing
    np.set_printoptions(precision=3, suppress=True)
    ''' MTCNN '''
    # MTCNN for re-doing the predictions, if I want to:
    device = init_device()
    mtcnn = MTCNN(image_size=384, select_largest=True, device=device, post_process=False, min_face_size=200)
    dir_imgs = '/mnt/sdb1/dp/rose_youtu_imgs_7k_(2)_14'
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
    # df = df[::100]
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
        # if blur_val < 20:
        #     plot_many(img, title=f'blur: {blur_val:.0f}')
        #    _ = input()
        #
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

    import seaborn as sns

    sns.set_style('whitegrid')
    sns.set_context('paper')

    plt.hist(blurs.blur, bins=100, range=[0, 500])
    plt.title('Variance of Laplacian')
    plt.tight_layout(pad=0)
    plt.show()

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

    # duplicate paths:
    dfdup = df[df.duplicated('path', keep=False)]
    dfdup.sort_values('path', inplace=True)

    # group by path
    dfdupg = dfdup.groupby('path')

    colors = ['r', 'g', 'b', 'y', 'm', 'c']

    videos_dir = '/mnt/sdb1/dp/rose_youtu_vids'
    kept = []
    # iterate over groups
    for path, group in dfdupg:
        # get original frame (luckily the first one in the video)
        try:
            vid = cv2.VideoCapture(join(videos_dir, group.source.iloc[0]))
            success, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = rotate(frame, find_rotation(mtcnn, frame))

            # re-run prediction on cropped image
            crop = Image.open(join(dir_imgs, path))
            c_bs, c_ps, c_ls = mtcnn.detect(crop, landmarks=True)
            c_bs, c_ps, c_ls = mtcnn.select_boxes(c_bs, c_ps, c_ls, crop, method=mtcnn.selection_method)
            c_ls = np.float32(c_ls[0])

            # check if landmarks are inside the image
            ls_min = c_ls.min(axis=0)
            ls_max = c_ls.max(axis=0)
            if ls_min[0] < 0 or ls_min[1] < 0 or \
                    ls_max[0] > crop.width or ls_max[1] > crop.height:
                print('landmarks outside of image, skipping...')
                continue
            # check if landmarks are close to the reference landmarks
            dst_landmarks = np.mean(np.linalg.norm(c_ls - ref_pts, axis=1))
            if dst_landmarks > 20:
                print(f'landmarks too far ({dst_landmarks:.2f}), skipping...')
                continue

            # which (if any) row is the correct one

            kept.append((path, None))


        except:
            # something failed, delete group and img?
            # or I just won't save the image into the clean version...
            pass

        f_bs, f_ps, f_ls = mtcnn.detect(frame, landmarks=True)

        # plot frame with all the existing detected bboxes
        plt.imshow(frame)
        for i, row in group.iterrows():
            # plot box
            box = row['box_orig'].astype(int)  # (4)
            plt.scatter(box[[0, 2]], box[[1, 3]], c=colors[i % len(colors)], label=i)
            bsize = np.linalg.norm(box[[2, 3]] - box[[0, 1]])
            print(i, bsize)
        plt.legend()
        plt.title(path)
        plt.tight_layout(pad=0)
        plt.show()

        # largest bbox
        sizes = [np.linalg.norm(box[[2, 3]] - box[[0, 1]]) for box in group['box_orig']]
        best_idx_by_box = np.argmax(sizes)

        # most accurate landmarks -- cannot compare with the saved ones, which are in the original frame?
        dists = [np.mean(np.linalg.norm(ls - ref_pts, axis=1)) for ls in group['landmarks']]
        best_idx_by_landmarks = np.argmin(dists)

        # plot frame with the bbox(es) of a new MTCNN prediction
        plt.imshow(frame)
        for i, b in enumerate(boxes):
            plt.scatter(b[[0, 2]], b[[1, 3]], c=colors[i % len(colors)], label=i)
            print(i, np.linalg.norm(b[[2, 3]] - b[[0, 1]]))
        plt.legend()
        plt.title(path)
        plt.tight_layout(pad=0)
        plt.show()

        # plot frame with the detected landmarks
        plt.imshow(frame)
        for i, r in group.iterrows():
            row_ls = r['landmarks']
            plt.scatter(row_ls[:, 0], row_ls[:, 1], c=colors[i % len(colors)], label=i)
            # print(i, np.linalg.norm(b[[2, 3]] - b[[0, 1]]))
        plt.legend()
        plt.title(path)
        plt.tight_layout(pad=0)
        plt.show()


        def plot_landmarks(img, landmarks, title=None):
            # plot frame with the detected landmarks
            plt.imshow(img)
            plt.scatter(landmarks[:, 0], landmarks[:, 1])
            if title:
                plt.title(title)
            plt.tight_layout(pad=0)
            plt.show()


        # plot crop
        plt.imshow(crop)
        plt.scatter(c_bs[0][[0, 2]], c_bs[0][[1, 3]], c='r')
        plt.show()

        # face height for all landmarks in the group
        from util_face import face_height

        heights = [face_height(ls) for ls in group['landmarks']]

        # plot image with all boxes from the same path

    ok_manual = []
    for p, _ in dfdupg:
        img = Image.open(join(dir_imgs, p))
        plt.imshow(img)
        plt.show()
        s = input()
        if len(s) > 0:
            ok_manual.append(p)
