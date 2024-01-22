import os
import cv2
import torch
import argparse
import warnings
import numpy as np
from utils import util
from time import time

warnings.filterwarnings("ignore")
OUT_DIR = "data/runs"
os.makedirs(OUT_DIR, exist_ok=True)

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

def calculate_box_distance(box1, box2):
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2

    distance = ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2) ** 0.5
    return distance

def merge_close_boxes(bboxes, distance_threshold):
    merged_bboxes = []
    while bboxes:
        base = bboxes.pop(0)
        i = 0
        while i < len(bboxes):
            # Check if the class ID is the same and the distance is within the threshold
            if base[4] == bboxes[i][4] and calculate_box_distance(base, bboxes[i]) < distance_threshold:
                # Create a merged box
                merged_box = [
                    min(base[0], bboxes[i][0]),
                    min(base[1], bboxes[i][1]),
                    max(base[2], bboxes[i][2]),
                    max(base[3], bboxes[i][3]),
                    base[4]  # Keep the same class_id
                ]
                base = merged_box
                bboxes.pop(i)
            else:
                i += 1
        merged_bboxes.append(base)
    return merged_bboxes


def process_model_output(output, args):
    output = util.non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.5, nc=args.num_classes)
    # print(f"output = util.non_max_suppression: {output}")
    holes = []  # List to store hole bounding boxes
    cracks = []
    for detection in output:
        detection = detection.view(-1, 6)
        # print(f"Detections: {detection}") 
        for x1, y1, x2, y2, conf, class_id in detection:
            y1 = max(y1 -10, 0)
            y2 = max(y2 -10, 0)
            # print(f"Conf: {conf.item()}, Class ID: {class_id.item()}")
            if conf >= args.threshold and class_id == 0:  # Check if it's a hole
                holes.append([x1, y1, x2, y2, class_id])
            if conf >= args.threshold and class_id == 1:  # Check if it's a hole
                cracks.append([x1, y1, x2, y2, class_id])
    print(f"Process_model_output holes:{len(holes)}")
    return holes , cracks


@torch.no_grad()
def run(args, data):
    model_path = './weights/allData/S/1280(v1+2_Val,v0+3+4_Train)Augmented+inversion10best.pt'
    model = torch.load(model_path, 'cuda')['model'].float().fuse()
    model.half()
    model.eval()

    original_batch_size = 28
    grid_batch_size = original_batch_size * 4  # 28 grid images per batch
    num_iterations = 1
    total_model_time = 0
    total_iteration_time = 0
    iteration_times = []
    start_collecting = False 

    for iteration in range(num_iterations):
        start_iteration = time()

        for i in range(0, len(data), original_batch_size):
            batch_paths = data[i:i + original_batch_size]
            all_holes = []  # Collect all hole detections for the entire batch
            all_cracks = []
            for path in batch_paths:
                original_image = preprocess_image(path, args.grid_size)
                grid_images = split_image_into_grid(original_image, args.grid_size)
                processed_images = []

                for img in grid_images:
                    x = torch.from_numpy(np.transpose(img, (2, 0, 1))).cuda().half() / 255.0
                    x = x.unsqueeze(0)  # Batch dimension

                    start_model = time()
                    output = model(x)
                    elapsed_model = time() - start_model

                    if start_collecting:
                        total_model_time += elapsed_model

                    holes , cracks = process_model_output(output, args)
                    all_holes.extend(holes)  # Collect holes from all grid images
                    all_cracks.extend(cracks)
                    processed_images.append(img)

                reassembled_image = reassemble_image_from_grid(processed_images, original_image.shape[:2], args.grid_size)

                # Merge close holes after reassembling
                merged_holes = merge_close_boxes(all_holes, args.distance_threshold)
                merged_cracks = merge_close_boxes(all_cracks, args.distance_threshold)

                green = (0, 255, 0) 
                red =  (0, 0, 255)
                # Draw merged holes on the reassembled image
                for hole in merged_holes:
                    x1, y1, x2, y2, class_id = hole
                    cv2.rectangle(reassembled_image, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
                
                # Draw merged cracks on the reassembled image
                for crack in merged_cracks:
                    x1, y1, x2, y2, class_id = crack
                    cv2.rectangle(reassembled_image, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)

                # Draw grid on the reassembled image
                height, width, _ = reassembled_image.shape
                x1, y1 = width // 2, height // 2
                cv2.line(reassembled_image, (x1, 0), (x1, height), (0, 255, 0), 2)
                cv2.line(reassembled_image, (0, y1), (width, y1), (0, 255, 0), 2)

                cv2.putText(reassembled_image, f'Holes: {len(merged_holes)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, green, 2)
                cv2.putText(reassembled_image, f'Cracks: {len(merged_cracks)}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, red, 2)

                # Save the reassembled image
                output_path = os.path.join(OUT_DIR, os.path.basename(path))
                cv2.imwrite(output_path, reassembled_image)

            elapsed_iteration = time() - start_iteration
            if start_collecting:
                iteration_times.append(elapsed_iteration)
                total_iteration_time += elapsed_iteration
                print(f'Iteration {iteration + 1}, Execution time: {elapsed_iteration} seconds')

        # Calculate and print average times
        average_model_time = total_model_time / (num_iterations * len(data) / original_batch_size)
        average_iteration_time = total_iteration_time / num_iterations
        print(f'Average MODEL execution time: {average_model_time} seconds')
        print(f'Average Inference time over {num_iterations} iterations: {average_iteration_time} seconds')
        print(f'Batch Processing: {grid_batch_size} Grid Images per Batch')


def get_data():
    ROOT = "data"
    dataset_loader = []
    for file in sorted(os.listdir(ROOT)):
        if file.endswith((".png",".jpg","tiff")):
            dataset_loader.append(os.path.join(ROOT, file))
    return dataset_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=2560, type=int)
    parser.add_argument('--num-classes', default=2, type=int) 
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--distance-threshold', default=5, type=int)
    parser.add_argument('--grid-size', nargs=2, default=[2, 2], type=int)
    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()
    data = get_data()
    run(args, data)

if __name__ == "__main__":
    main()
