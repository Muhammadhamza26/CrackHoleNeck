import os
import cv2
import torch
import argparse
import warnings
import numpy as np
from utils import util
from time import time
from jaistreaming import Camera

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
def run(args):
    # Load the model
    model_path = './weights/allData/S/1280(v1+2_Val,v0+3+4_Train)Augmented+inversion10best.pt'
    model = torch.load(model_path, 'cuda')['model'].float().fuse()
    model.half()
    model.eval()

    connection_ID = ['169.254.17.193'] 
    camera = Camera(args, connection_ID)

    for frame in camera.stream_frames(args):
        # Preprocess the frame
        processed_frame = preprocess_image(frame, args.grid_size)

        # Split frame into grid
        grid_images = split_image_into_grid(processed_frame, args.grid_size)

        all_holes = []  # To store hole detections
        all_cracks = [] # To store crack detections

        # Process each grid image
        for img in grid_images:
            x = torch.from_numpy(np.transpose(img, (2, 0, 1))).cuda().half() / 255.0
            x = x.unsqueeze(0)  # Add batch dimension

            # Model inference
            output = model(x)

            # Process model output
            holes, cracks = process_model_output(output, args)
            all_holes.extend(holes)
            all_cracks.extend(cracks)

        # Reassemble the frame from grid images
        reassembled_frame = reassemble_image_from_grid(grid_images, frame.shape[:2], args.grid_size)

        # Merge close boxes
        merged_holes = merge_close_boxes(all_holes, args.distance_threshold)
        merged_cracks = merge_close_boxes(all_cracks, args.distance_threshold)

        # Draw detections on the frame
        for hole in merged_holes:
            x1, y1, x2, y2, class_id = hole
            cv2.rectangle(reassembled_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        for crack in merged_cracks:
            x1, y1, x2, y2, class_id = crack
            cv2.rectangle(reassembled_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Processed Frame', reassembled_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources after exiting the loop
    camera.disconnect_streams()
    cv2.destroyAllWindows()


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
    run(args)

if __name__ == "__main__":
    main()
