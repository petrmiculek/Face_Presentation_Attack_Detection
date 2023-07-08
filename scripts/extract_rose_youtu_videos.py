# stdlib
from glob import glob
import os
import sys
from os.path import join
import zipfile
import io
import argparse

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
parser.add_argument('--input_path', type=str, default='/mnt/sdb1/dp/rose_youtu_vids')
# parser.add_argument('--matching_pattern', type=str, default='*.mp4')
parser.add_argument('--output_path', type=str, default='/mnt/sdb1/dp/rose_youtu_crops')

if __name__ == '__main__':
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print(f'Running: {__file__}\nIn dir: {os.getcwd()}')
    print('Args:', ' '.join(sys.argv))

    device = init_device()
    mtcnn = MTCNN(image_size=384,  # TODO 384?
                  select_largest=False, device='cuda', margin=40,
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
    filenames = glob(matching_pattern)[0:2]
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
                if i_frame % filter_rate != 0 or not success:
                    continue

                # Accumulate batch
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(Image.fromarray(frame))

                ''' Save full frame - unused '''
                # r, b = cv2.imencode(dot_ext, frame)
                # zip_io.writestr(f"{filename_base}_full_{i}.{dot_ext}", b.tobytes())

                ''' Batch Predict '''
                if len(batch) >= batch_size or i_frame == v_len - 1:
                    # crops = mtcnn(batch)
                    boxes, probs, landmarks = mtcnn.detect(batch, landmarks=True)
                    if not mtcnn.keep_all:
                        save_path = None  # TODO can try saving through this
                        return_prob = False
                        boxes, probs, landmarks = mtcnn.select_boxes(
                            boxes, probs, landmarks, batch, method=mtcnn.selection_method)
                    crops = mtcnn.extract(batch, boxes, save_path=None)

                    for i, i_retro in enumerate(idxs):
                        frame = batch[i]
                        if (face_crop := crops[i]) is None:
                            continue
                        # save cropped face
                        crop_filename = f"{filename_base}_crop_{i_retro}{dot_ext}"
                        face_npy = face_crop.permute(1, 2, 0).numpy()
                        # r, b = cv2.imencode(dot_ext, face_npy)
                        # zip_io.writestr(crop_filename, b.tobytes())
                        cv2.imwrite(join(output_path, crop_filename),
                                    cv2.cvtColor(face_npy, cv2.COLOR_RGB2BGR))  # alternative saving
                        # cv2.imwrite(join(output_path, crop_filename), face_npy)  # alternative saving

                        data.append({'source': vid_filename, 'img': crop_filename,
                                     'frame': frame, 'face': face_crop,  # <- temporary
                                     'box': boxes[i], 'prob': probs[i], 'landmark': landmarks[i]})
                    batch = []
                    idxs = []
                    pbar.set_description(desc=f'{i}/{v_len}')
            # end of video

            ''' Write to disk '''
            # with open(output_path_zip, "wb") as f:
            #     f.write(byte_io.getvalue())

    # end of all videos

    df = pd.DataFrame(data)
    # Visualize
    if False:
        for bs, probs, landmarks in zip(boxes, probs, landmarks):
            fig, ax = plt.subplots()  # figsize=(16, 12)
            ax.imshow(frame)
            ax.axis('off')

            for b, l in zip(bs, landmarks):
                ax.scatter(*np.meshgrid(b[[0, 2]], b[[1, 3]]))
                ax.scatter(l[:, 0], l[:, 1], s=8)
            fig.tight_layout()
            fig.show()

        frame = Image.fromarray(frame)

        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        plt.axis('off')
