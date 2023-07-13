# stdlib
import os
import sys
from os.path import join

# fix for local import problems
cwd = os.getcwd()
sys.path.extend([cwd] + [join(cwd, d) for d in os.listdir() if os.path.isdir(d)])

# external
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

# local
from util_torch import init_device
from util import plot_many  # debugging

if __name__ == '__main__':
    # human-readable numpy float printing
    np.set_printoptions(precision=3, suppress=True)

    ''' MTCNN '''
    # MTCNN for re-doing the predictions, if I want to:
    device = init_device()
    mtcnn = MTCNN(image_size=384, select_largest=False, device=device, post_process=False)

    ''' Paths '''
    dir_vid = '/mnt/sdb1/dp/rose_youtu_vids'
    dir_imgs = '/mnt/sdb1/dp/rose_youtu_single'
    df = pd.read_pickle(join(dir_imgs, 'data.pkl'))

    dir_out = '/mnt/sdb1/dp/rose_youtu_single_aligned'
    os.makedirs(dir_out, exist_ok=False)

    dfg = df.groupby('source')
    sources = list(dfg.groups.keys())
    vid_name = sources[0]

    ''' Reference landmarks '''
    """
    Following code section adapted from: Jakub Špaňhel
    # reference model uses input size 336x336 (112x112 * 2 + 112)
    5 IPT points: eye left, eye right, nose, mouth left, mouth right
    """
    scale = 3.25
    enlarge = 384 - 112 * scale

    # scale = 2.0
    # enlarge = 384 - 112 * scale

    # IPT points coordinates are for image 112x112 pixels
    ref_pts = np.float32([[38.0, 38.0],
                          [74.0, 38.0],
                          [56.0, 58.0],
                          [40.0, 76.0],
                          [72.0, 76.0]]) * scale + enlarge / 2

    img_size = (np.array([112, 112]) * scale + enlarge).astype(int)  # (width, height) = x, y

    border_mode = cv2.BORDER_CONSTANT  # cv2.BORDER_REFLECT <- caused reflection artifacts, unused
    reflecting = border_mode == cv2.BORDER_REFLECT

    vid_info = []
    for video_name_ in tqdm(sources):
        vid_obj_ = cv2.VideoCapture(join(dir_vid, video_name_))
        vid_info.append({
            'name': video_name_,
            'orientation_meta': vid_obj_.get(cv2.CAP_PROP_ORIENTATION_META),
            'orientation_auto': vid_obj_.get(cv2.CAP_PROP_ORIENTATION_AUTO),
            'dim': (vid_obj_.get(cv2.CAP_PROP_FRAME_WIDTH), vid_obj_.get(cv2.CAP_PROP_FRAME_HEIGHT))
        })

    vid_info = pd.DataFrame(vid_info)
    ''' Process videos '''
    sus = []  # manually marked problematic videos: w=wrong, f=frame, b=bbox
    n_skipped = 0
    # for vid_name in tqdm(sources):
    i_vid = -1
    while i_vid < len(sources):
        i_vid += 1
        vid_name = sources[-i_vid]
        ''' Load video '''
        vid_path = join(dir_vid, vid_name)
        vid = cv2.VideoCapture(vid_path)

        rot_meta = vid.get(cv2.CAP_PROP_ORIENTATION_META)

        rows = dfg.get_group(vid_name)
        names_to_keep = rows['path'].values
        bboxs = rows['box'].values
        landmarks = rows['landmarks'].values
        dims = rows['dim_orig'].values  # (height, width), whereas bboxs and landmarks are (w, h)

        # normally there would be more frames per video, here the dataset contains just one.
        # loop over frames in video is skipped here.
        # read last int of filename as frame number
        idx_video = [int(n.split('_')[-1].split('.')[0]) for n in names_to_keep]
        last_idx = max(idx_video)

        ''' Get frames '''
        frames = []
        for i in range(1000):
            print('.', end='')
            success, frame = vid.read()
            # plt.imshow(frame); plt.show()
            if not success or i > last_idx:
                break

            if i in idx_video:
                frames.append(frame)

        assert len(frames) == len(idx_video)

        ''' Get faces '''
        for i, frame in enumerate(frames):
            n_skipped += 1
            dim_frame = dims[i][::-1]  # (width, height) = x, y
            box = bboxs[i]
            filename = names_to_keep[i]

            # box outside the frame
            if box[0] < 0 or box[1] < 0 or box[2] > dim_frame[0] or box[3] > dim_frame[1]:
                print(f'box out of bounds: {box}')
                continue

            lms = landmarks[i].astype(np.float32)
            # frame is HWC, BGR at first
            if rot_meta == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rot_meta == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot_meta == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            h, w = frame.shape[:2]  # h, w
            if w > h:
                print(f'{vid_name}: w > h')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            eyel, eyer, _, mouthl, mouthr = lms

            ''' Compute affine transformation to crop the face '''
            # corresponding points are 1) left eye, 2) right eye, 3) center of mouth
            affine_from = np.float32([eyer, eyel, (mouthl + mouthr) // 2])
            affine_to = np.float32([ref_pts[1], ref_pts[0], (ref_pts[3] + ref_pts[4]) // 2])
            transform_mouth = cv2.getAffineTransform(affine_from, affine_to)
            transform_3x3 = np.float32(np.r_[transform_mouth, [[0, 0, 1]]])

            if reflecting:
                # check how close to the edge is the bbox, if too close, skip
                transform_inverse = np.linalg.inv(transform_3x3)

                # reverse-project corners of the cropped image to the original image
                corners = np.float32([[0, 0], [img_size[0], 0], [0, img_size[1]], [img_size[0], img_size[1]]])
                corners_inv = cv2.perspectiveTransform(corners.reshape(1, -1, 2), transform_inverse)[
                    0]  # crop-corners projected to image

                # corner to original frame border distance must not be higher than distance(leftmost_keypoint, edge)
                # check for left and top
                e = 30  # safety margin
                lm_to_left, lm_to_top = lms.min(axis=0) - [0, 0] - [e, e]  # landmarks - frame corner - safety margin
                lm_to_right, lm_to_bottom = dim_frame - lms.max(axis=0) - [e,
                                                                           e]  # frame corner - landmarks + safety margin

                inv_left = (corners_inv[0, 0] + corners_inv[2, 0]) / 2
                inv_top = (corners_inv[0, 1] + corners_inv[1, 1]) / 2
                inv_right = (corners_inv[1, 0] + corners_inv[3, 0]) / 2
                inv_bottom = (corners_inv[2, 1] + corners_inv[3, 1]) / 2

                reflects_left = inv_left < 0 - lm_to_left
                reflects_top = inv_top < 0 - lm_to_top
                reflects_right = inv_right > dim_frame[0] + lm_to_right
                reflects_bottom = inv_bottom > dim_frame[1] + lm_to_bottom

                if any([reflects_left, reflects_top, reflects_right, reflects_bottom]):
                    print(f'box too close to edge: {box}')
                    continue

            if True:
                batch = frame[None, ...]
                boxes2, probs2, landmarks2 = mtcnn.detect(batch, landmarks=True)
                boxes2, probs2, landmarks2 = mtcnn.select_boxes(boxes2, probs2, landmarks2, batch)
                crops2 = mtcnn.extract(batch, boxes2, save_path=None)
                if crops2 not in [[None], None]:
                    boxes2 = boxes2[0, 0]
                    landmarks2 = landmarks2[0, 0]
                    # are boxes2 inside the frame?
                    boxes_or_landmarks_out_of_bounds = False
                    if boxes2[0] < 0 or boxes2[1] < 0 or boxes2[2] > dim_frame[0] or boxes2[3] > dim_frame[1]:
                        print(f'box out of bounds: {boxes2}')
                        boxes_or_landmarks_out_of_bounds = True
                    # ditto landmarks
                    if any([*(landmarks2.min(axis=0) < [0, 0]), *(landmarks2.max(axis=0) > dim_frame)]):
                        print(f'landmarks out of bounds:\n', landmarks2)
                        boxes_or_landmarks_out_of_bounds = True
                    if boxes_or_landmarks_out_of_bounds:
                        continue
                else:
                    print(f'no face detected for {vid_name}: {idx_video[i]}')

            ''' Apply transformation '''
            box_vec2 = box.reshape(1, -1, 2).astype(np.float32)
            box_cropped = cv2.perspectiveTransform(box_vec2, transform_3x3)[0]  # box projected to crop
            crop = cv2.warpAffine(frame, transform_mouth, img_size, borderMode=border_mode, flags=cv2.INTER_AREA)

            ''' Save cropped image '''
            cv2.imwrite(join(dir_out, filename), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            # TODO save box_t, and landmarks (only once)

            plt.imshow(crop)
            plt.title(f'{vid_name}: {idx_video[i]}')
            plt.scatter(ref_pts[:, 0], ref_pts[:, 1])
            plt.scatter(*np.meshgrid(box_cropped[:, 0], box_cropped[:, 1]), marker='x', c='r')
            plt.axis('off')
            plt.tight_layout(pad=0.1)
            plt.show()
            plt.close()
            # wait for input
            n_skipped -= 1
            s = input()
            if len(s) > 0:
                sus.append({'vid_name': vid_name,
                            'idx_video': idx_video[i],
                            'box_orig': box,
                            'box_crop': box_cropped,
                            'landmarks': lms,
                            'frame': frame,
                            'crop': crop,
                            'note': s,
                            })

    print(f'{n_skipped} frames skipped')
    if False:
        # TODO BORDER_REFLECT OK?  [func]
        # borders = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101,
        #            cv2.BORDER_WRAP, cv2.BORDER_TRANSPARENT]
        # flags = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        # transform_nose = cv2.getAffineTransform(lms[:3], ref_pts[:3])
        # image_nose = cv2.warpAffine(frame, transform_nose, (384, 384), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_AREA)
        # plot_many(frame, crop, image_nose)

        # compare reference points for transformation
        if False:
            # compare matrices
            plt.subplots(1, 2)
            plt.subplot(121)
            plt.imshow(crop)
            plt.title('mouth center')
            plt.scatter(ref_pts[:, 0], ref_pts[:, 1])
            # plt.axis('off')
            plt.subplot(122)
            plt.imshow(image_nose)  # commented out
            plt.title('nose')
            plt.scatter(ref_pts[:, 0], ref_pts[:, 1])
            plt.axis('off')
            plt.tight_layout(pad=0.1)
            plt.show()

        # plot bbox, landmarks
        if False:
            half_margin = 140 // 2

            minus_plus = np.array([0, 0]) * half_margin
            fig, ax = plt.subplots()  # figsize=(16, 12)
            ax.imshow(frame)
            ax.scatter(lms[:, 0], lms[:, 1])
            ax.scatter(*np.meshgrid(box[[0, 2]] + minus_plus, box[[1, 3]] + minus_plus))
            ax.scatter(corners_inv[:, 0], corners_inv[:, 1])
            # fig.suptitle(f'{minus_plus}')
            fig.show()
            if False:
                # ax.axis('off')
                for l in lms:
                    ax.scatter(l[0], l[1], s=8)
                fig.tight_layout()
                fig.show()  # originally here
