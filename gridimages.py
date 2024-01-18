## spliting images into grid and save

import json
import os
import cv2

# Assuming label files are in JSON format with bbox coordinates
def read_labels(label_path, image_size):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x1 = (x_center - width / 2) * image_size[1]
            y1 = (y_center - height / 2) * image_size[0]
            x2 = (x_center + width / 2) * image_size[1]
            y2 = (y_center + height / 2) * image_size[0]
            labels.append({'bbox': [x1, y1, x2, y2], 'class': class_id})
    return labels

def adjust_bboxes_for_grids(grids, labels, grid_size):
    adjusted_labels = []
    rows, cols = grid_size
    grid_h, grid_w = grids[0].shape[0], grids[0].shape[1]

    for i in range(rows):
        for j in range(cols):
            grid_labels = []
            for label in labels:
                # Adjust bbox coordinates for current grid
                x1, y1, x2, y2 = label['bbox']
                adjusted_x1 = max(x1 - j * grid_w, 0)
                adjusted_y1 = max(y1 - i * grid_h, 0)
                adjusted_x2 = min(x2 - j * grid_w, grid_w)
                adjusted_y2 = min(y2 - i * grid_h, grid_h)

                # Check if bbox is within the current grid
                if adjusted_x1 < grid_w and adjusted_y1 < grid_h and adjusted_x2 > 0 and adjusted_y2 > 0:
                    grid_labels.append({'bbox': [adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2], 'class': label['class']})

            adjusted_labels.append(grid_labels if grid_labels else None)

    return adjusted_labels

def save_grid_and_label(grid, label, grid_idx,img_file, output_dir, grid_size):
    base_img_name = os.path.splitext(img_file)[0]
    img_name = f"grid_{grid_idx}_{base_img_name}.png"
    label_name = f"grid_{grid_idx}_{base_img_name}.txt"

    # Ensure output directories exist
    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_img_dir, img_name), grid)
    print(f'saving {img_name}')
    with open(os.path.join(output_label_dir, label_name), 'w') as f:
        for lbl in label:
                x1, y1, x2, y2 = lbl['bbox']
                class_id = lbl['class']

                # Convert bbox coordinates back to normalized format relative to the grid
                grid_w, grid_h = grid.shape[1], grid.shape[0]
                x_center = ((x1 + x2) / 2) / grid_w
                y_center = ((y1 + y2) / 2) / grid_h
                width = (x2 - x1) / grid_w
                height = (y2 - y1) / grid_h

                # Save the normalized bbox coordinates to the label file
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def preprocess_image(image_path, grid_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_path} not found or format not supported")

    h, w = image.shape[:2]
    rows, cols = grid_size
    pad_h = (rows - h % rows) % rows
    pad_w = (cols - w % cols) % cols

    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h // 2, pad_h - pad_h // 2,
                                   pad_w // 2, pad_w - pad_w // 2,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image


def split_image_into_grid(image, grid_size):
    rows, cols = grid_size
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols

    split_images = []
    for i in range(rows):
        for j in range(cols):
            start_row, start_col = i * grid_h, j * grid_w
            end_row, end_col = start_row + grid_h, start_col + grid_w
            split_images.append(image[start_row:end_row, start_col:end_col])
    return split_images

def preprocess_and_split_all_images(images_dir, labels_dir, grid_size, output_dir):
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))

    # Iterate over image_files and label_files
    for img_file, lbl_file in zip(image_files, label_files):
        try:
            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, lbl_file)

            print(f'Processing {img_file} and {lbl_file}...')
            image = preprocess_image(image_path, grid_size)
            h, w = image.shape[:2]
            grids = split_image_into_grid(image, grid_size)
            labels = read_labels(label_path, (h, w))

            adjusted_labels = adjust_bboxes_for_grids(grids, labels, grid_size)

            for idx, (grid, adjusted_label) in enumerate(zip(grids, adjusted_labels)):
                if adjusted_label:  # Filter out grids without bboxes
                    save_grid_and_label(grid, adjusted_label, idx, img_file, output_dir, grid_size)

        except Exception as e:
            print(f'An error occurred while processing {img_file}: {e}')


# Main function call
images_dir = "/hdd/KIA_Smart_Factory_Crack_Hole_detection/hamza/v1/V1_C&H_Shiraz/grid/images/train/"
labels_dir = "/hdd/KIA_Smart_Factory_Crack_Hole_detection/hamza/v1/V1_C&H_Shiraz/grid/labels/train/"
grid_size = (2, 2)
output_dir = "/hdd/KIA_Smart_Factory_Crack_Hole_detection/hamza/v1/V1_C&H_Shiraz/grid/out/"
os.makedirs(output_dir, exist_ok=True)
preprocess_and_split_all_images(images_dir, labels_dir, grid_size, output_dir)
