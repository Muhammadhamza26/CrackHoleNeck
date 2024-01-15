import os
import cv2
import torch
import argparse
import warnings
import numpy as np
from utils import util

warnings.filterwarnings("ignore")
OUT_DIR = "././data/out"

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


def reassemble_image_from_grid(split_images, original_image_size, grid_size):
    rows, cols = grid_size
    assert len(split_images) == rows * cols, f"Mismatch in the number of split images and grid size: expected {rows * cols}, got {len(split_images)}"
    h, w = original_image_size
    grid_h, grid_w = h // rows, w // cols

    reassembled_image = np.zeros((h, w, 3), dtype=np.uint8)
    assert len(split_images) == rows * cols, "Mismatch in the number of split images and grid size"

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(split_images):
                start_row, start_col = i * grid_h, j * grid_w
                reassembled_image[start_row:start_row + grid_h, start_col:start_col + grid_w] = split_images[idx]

    return reassembled_image


@torch.no_grad()
def run(args, data):
    model_path = './../weights/allData/S/1280(v1+2_Val,v0+3+4_Train)Augmented+inversion10best.pt'
    model = torch.load(model_path, 'cuda')['model'].float().fuse()
    model.half()
    model.eval()

    total_holes = 0
    total_cracks = 0

    for path in data:
            original_image = preprocess_image(path, args.grid_size)
            split_images = split_image_into_grid(original_image, args.grid_size)
            processed_images = []

            for img in split_images:
                # Prepare the image for the model
                x = torch.from_numpy(img.transpose((2, 0, 1))).cuda().half() / 255.0
                x = x.unsqueeze(0)  # Add batch dimension

                # Model inference
                output = model(x)

                # Process the model output
                processed_img, num_holes, num_cracks = process_model_output(output, img, args)
                processed_images.append(processed_img)
                total_holes += num_holes
                total_cracks += num_cracks

            reassembled_image = reassemble_image_from_grid(processed_images, original_image.shape[:2], args.grid_size)

    # Print the total counts on the reassembled image
    cv2.putText(reassembled_image, f'Total Holes: {total_holes}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(reassembled_image, f'Total Cracks: {total_cracks}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the reassembled image with the counts printed on it
    output_path = os.path.join(OUT_DIR, f"processed_{os.path.basename(path)}")
    cv2.imwrite(output_path, reassembled_image)

def process_model_output(output, img, args):
    # Apply non-max suppression to filter the output
    output = util.non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.5, nc=args.num_classes)
    
    # Initialize counters for holes and cracks
    num_holes = 0
    num_cracks = 0
    
    # Process each detection
    for detection in output:
        for x1, y1, x2, y2, conf, class_id in detection:
            if conf.item() >= args.threshold:
                
                ## manually pushing bboxes up
                y1 = max(y1 - 10, 0)
                y2 = max(y2 - 10, 0)

                if class_id.item() == 0:
                    num_holes += 1
                elif class_id.item() == 1:
                    num_cracks += 1

                color = (0, 255, 0) if class_id.item() == 0 else (0, 0, 255)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return img, num_holes, num_cracks

def get_data():
    ROOT = "./../data/"
    dataset_loader = []
    for file in sorted(os.listdir(ROOT)):
        if file.endswith((".png",".jpg")):
            dataset_loader.append(os.path.join(ROOT, file))
    return dataset_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=2560, type=int)
    parser.add_argument('--num-classes', default=2, type=int) 
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--grid-size', nargs=2, default=[2, 2], type=int)
    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()
    data = get_data()
    run(args, data)

if __name__ == "__main__":
    main()
