# stdlib
from glob import glob
import os
import sys
from os.path import join
import zipfile
import io
import argparse

# fix for local import problems
sys.path.extend([os.getcwd()] + [d for d in os.listdir() if os.path.isdir(d)])

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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, default='/mnt/sdb1/dp/rose_youtu_vids')
parser.add_argument('-o', '--output_path', type=str, default='/mnt/sdb1/dp/rose_youtu_crops')

if __name__ == '__main__':
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))

    ''' Face Detector - MTCNN'''
    device = init_device()
    mtcnn = MTCNN(image_size=384,
                  select_largest=False, device='cuda', margin=140,
                  post_process=False,  # TODO check effect
                  # keep_all=True,  # TODO check effect
                  )
    ''' Setup '''
    input_path, output_path = args.input_path, args.output_path
    matching_pattern = '*.mp4'
    # output_path_prefix_zip = '/mnt/sdb1/dp/rose_youtu_imgs_zip'
    cwd_prev = os.getcwd()
    os.chdir(input_path)
    for p in [output_path]:  # output_path_prefix_zip
        if not os.path.exists(p):
            os.makedirs(p)
            print(f'Created dir: {p}')

    # glob_pattern = os.path.join(input_path, matching_pattern)
    filenames = glob(matching_pattern)[3:4]
    vid_filename = filenames[0]
    dot_ext = '.jpg'
    batch_size = 32
    data = []

    ''' Processing videos '''
    pbar = tqdm(filenames)
    for vid_filename in pbar:
        filename_base = vid_filename[: -len(".mp4")]
        v_cap = cv2.VideoCapture(vid_filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # output_path_zip = join(output_path_prefix_zip, filename_base) + ".zip"

        batch = []
        idxs = []
        ''' Write to zip (in memory) '''
        # byte_io = io.BytesIO()
        # zip_io = zipfile.ZipFile(byte_io, "w")
        # with io.BytesIO() as byte_io, zipfile.ZipFile(byte_io, "w") as zip_io:
        filter_rate = v_len // 100  # get around 100 frames. More because of rounding, less because of non-face frames.
        if True:
            for i_frame in range(v_len):  # , leave=False
                # Load frame
                success, frame = v_cap.read()
                last = i_frame == v_len - 1
                skipped = i_frame % filter_rate != 0
                if not last and skipped or not success:
                    continue

                # Accumulate batch
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(Image.fromarray(frame))
                idxs.append(i_frame)

                # Batch Predict
                if len(batch) >= batch_size or last:
                    mtcnn.margin = 0

                    boxes, probs, landmarks = mtcnn.detect(batch, landmarks=True)
                    if not mtcnn.keep_all:
                        boxes, probs, landmarks = mtcnn.select_boxes(boxes, probs, landmarks,
                                                                     batch, method=mtcnn.selection_method)
                    # box format: [x0, y0, x1, y1]
                    # landmark format: [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4] = [left_eye, right_eye, nose, left_mouth, right_mouth]
                    boxes = boxes.astype(float).round(0).astype(int)
                    landmarks = landmarks.astype(float).round(0).astype(int)
                    crops = mtcnn.extract(batch, boxes, save_path=None)
                    # Per-image saving
                    for i, i_retro in enumerate(idxs):
                        frame = batch[i]
                        if (face_crop := crops[i]) is None:
                            continue
                        # save cropped face
                        crop_filename = f"{filename_base}_crop_{i_retro}{dot_ext}"
                        path_crop = join(output_path, crop_filename)
                        face_npy = face_crop.permute(1, 2, 0).numpy()
                        cv2.imwrite(path_crop, cv2.cvtColor(face_npy, cv2.COLOR_RGB2BGR))  # alternative saving
                        data.append({'source': vid_filename, 'path': path_crop,
                                     'box': boxes[i], 'landmarks': landmarks[i]})
                    batch = []
                    idxs = []
                    pbar.set_description(desc=f'frame({i_frame}/{v_len})')
            # end of video
    # end of all videos

    df = pd.DataFrame(data)

    # save df as pickle
    df.to_pickle(join(output_path, 'data.pkl'))

    # Visualize
    if False:
        half_margin = mtcnn.margin // 2
        minus_plus = np.array([-1, 1]) * half_margin
        for ls, frame in zip(landmarks, batch):
            fig, ax = plt.subplots()  # figsize=(16, 12)
            ax.imshow(frame)
            # ax.axis('off')

            for l in ls:
                ax.scatter(*np.meshgrid(b[[0, 2]] + minus_plus, b[[1, 3]] + minus_plus))
                h, w = b[[1, 3]] + minus_plus - b[[0, 2]] + minus_plus
                print(h, w)
                ax.scatter(l[:, 0], l[:, 1], s=8)
            fig.tight_layout()
            fig.show()
            break

        frame = Image.fromarray(frame)

        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        plt.axis('off')

    # Explore margin parameter
    if False:
        from util import plot_many

        crops_margins = []
        margins = []
        for margin in range(0, 240, 20):
            mtcnn.margin = margin
            crops = mtcnn.extract(batch, boxes, save_path=None)[0]

            crops_margins.append(crops / 255)
            margins.append(margin)

        plot_many(crops_margins, titles=margins, title='Margins')
