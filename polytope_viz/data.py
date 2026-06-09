import cv2
import numpy as np

from .coordinates import pixel_to_model_coords


def preprocess_grayscale_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not open image: {image_path}")
    image = image.astype(np.float32) / 255.0
    height, width = image.shape
    data = []
    for i in range(height):
        for j in range(width):
            x, y = pixel_to_model_coords(j, i, width, height)
            data.append((x, y, image[i, j]))
    return np.array(data), (height, width)


def preprocess_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    data = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_norm = frame_idx / frame_count
        for i in range(height):
            for j in range(width):
                x, y = pixel_to_model_coords(j, i, width, height)
                data.append((x, y, t_norm, gray_frame[i, j]))
        frame_idx += 1
    cap.release()
    return np.array(data), (height, width)


def preprocess_classifier_image(image_path: str, logger=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    data = []
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            if r == 255 and g == 0 and b == 0:
                x, y = pixel_to_model_coords(j, i, width, height)
                data.append((x, y, 0))
            elif r == 0 and g == 255 and b == 0:
                x, y = pixel_to_model_coords(j, i, width, height)
                data.append((x, y, 1))
            elif r == 0 and g == 0 and b == 255:
                x, y = pixel_to_model_coords(j, i, width, height)
                data.append((x, y, 2))
    if not data:
        raise ValueError(f"No red, green, or blue pixels found in image: {image_path}")
    if logger:
        logger.info(f"Found {len(data)} colored pixels in image")
    return np.array(data), (height, width)


def sample_data(data, train_size: int, val_size: int):
    if train_size + val_size > len(data):
        raise ValueError("Requested training + validation size exceeds dataset size.")
    np.random.shuffle(data)
    return data[:train_size], data[train_size:train_size + val_size]
