import numpy as np
import cv2


def rotate(img, angle):
    """ Rotate image by angle. """
    # if angle == 0: pass
    if angle == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def find_rotation(mtcnn, img, threshold=0.9):
    """ Select best rotation for image. """
    versions = []
    rotations = [0, 90, 180, 270]
    for rot in rotations:
        versions.append(rotate(img, rot))

    # predict for all
    # batch = np.stack(versions)
    b, probs_raw, _ = mtcnn_predict_batched(mtcnn, versions)

    # keep max prob for each image
    probs = []
    for p in probs_raw:
        p = p[0]
        if p is None:
            p = 0
        else:
            p = np.max(p)
        probs.append(p)

    # if all none
    if np.max(probs) <= threshold:
        return None

    # select best prob
    best_idx = np.argmax(probs)
    return rotations[best_idx]


def face_height(landmarks):
    """ Get face height from landmarks. """
    eyec = (landmarks[0] + landmarks[1]) / 2
    mouthc = (landmarks[3] + landmarks[4]) / 2
    landmarks_height = (mouthc - eyec)[1]
    return landmarks_height


def get_ref_landmarks(scale=3.25, margin=None, img_shape_verify=(384, 384)):
    """ Get IPT reference landmarks for face alignment.
    [left eye, right eye, nose, left mouth, right mouth]
    defined for a 112x112 image, scale/pad accordingly.

    3 points chosen for the affine transformation:
    - right eye, left eye, center of mouth

    :param scale: scale of image
    :param margin: margin of image (total == both sides)
    :param img_shape_verify: image shape
    :return: IPT reference landmarks,
             derived alignment landmarks
    """
    if margin is None:
        margin = img_shape_verify[0] - 112 * scale
    assert (112 * scale + margin) == img_shape_verify[0], \
        f'scale and margin must be chosen to match image size {img_shape_verify[0]}'
    ipt_pts = np.float32([[38.0, 38.0], [74.0, 38.0], [56.0, 58.0],
                          [40.0, 76.0], [72.0, 76.0]]) * scale + margin / 2
    return ipt_pts


def select_landmarks_points(ipt_pts):
    """
    Choose 3 points for the affine transformation.
    corresponding points are 1) left eye, 2) right eye, 3) center of mouth
    :param ipt_pts: IPT reference landmarks
    :return: 3 points for the affine transformation
    """
    return np.float32([ipt_pts[1], ipt_pts[0], (ipt_pts[3] + ipt_pts[4]) // 2])


def get_align_transform(landmarks_from, landmarks_to):
    """ Get affine transformation matrix from points.
    :param landmarks_from: source points [5]
    :param landmarks_to: target points [5]
    :return: 2x3 affine transformation matrix
    """
    ''' Compute affine transformation to crop the face '''
    from3 = select_landmarks_points(landmarks_from)
    to3 = select_landmarks_points(landmarks_to)
    return cv2.getAffineTransform(from3, to3)


def transform_bbox(box_orig, transform2x3):
    """
    Transform bounding box.
    :param box_orig: original bounding box [4]
    :param transform2x3: affine transformation matrix
    :return: transformed bounding box [-1, 2]
    """
    transform_3x3 = np.float32(np.r_[transform2x3, [[0, 0, 1]]])
    box_orig_vec2 = box_orig.reshape(1, -1, 2).astype(np.float32)
    box_cropped = cv2.perspectiveTransform(box_orig_vec2, transform_3x3)[0]  # box projected to crop
    return box_cropped


def mtcnn_predict_batched(mtcnn, batch):
    """ Predict faces in a batch of images.
    When image shapes are not all the same in the batch, error is thrown.
    Predict one by one in that case.
    """
    try:
        boxes, probs, landmarks = mtcnn.detect(batch, landmarks=True)
    except Exception as e:
        # silent, fallback: predict one by one
        boxes, probs, landmarks = [], [], []
        for i, frame in enumerate(batch):
            try:
                box, prob, lms = mtcnn.detect(frame, landmarks=True)
            except Exception as e2:
                box, prob, lms = None, None, None
            finally:
                boxes.append(box)
                probs.append(prob)
                landmarks.append(lms)
        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)
        landmarks = np.array(landmarks, dtype=object)

    return boxes, probs, landmarks
