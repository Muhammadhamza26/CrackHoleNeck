import os
import cv2
import torch
import argparse
import warnings
import numpy as np
from utils import util
from time import time

warnings.filterwarnings("ignore")

def preprocess_image(image_path, grid_size):
    """Load and pad image to make its dimensions divisible by grid size."""
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
    """Split image into smaller grids based on specified grid size."""
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
    h, w = original_image_size
    grid_h, grid_w = h // rows, w // cols

    reassembled_image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            start_row, start_col = i * grid_h, j * grid_w
            reassembled_image[start_row:start_row + grid_h, start_col:start_col + grid_w] = split_images[idx]

# # Draw grid lines
    # # Vertical lines
    # for i in range(1, cols):
    #     x = i * grid_w
    #     cv2.line(reassembled_image, (x, 0), (x, h), (0, 255, 0), 2)  # Green line

    # # Horizontal lines
    # for i in range(1, rows):
    #     y = i * grid_h
    #     cv2.line(reassembled_image, (0, y), (w, y), (0, 255, 0), 2)  # Green line

    return reassembled_image


def calculate_box_distance(box1, box2):
    """Calculate the Euclidean distance between the centers of two bounding boxes."""
    center_x1, center_y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    center_x2, center_y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2) ** 0.5


def merge_global_bboxes(global_bboxes, distance_threshold):
    """Merge global bounding boxes based on a distance threshold."""
    merged_bboxes = []
    while global_bboxes:
        base = global_bboxes.pop(0)
        i = 0
        while i < len(global_bboxes):
            if calculate_box_distance(base, global_bboxes[i]) < distance_threshold:
                merged_box = [
                    min(base[0], global_bboxes[i][0]), min(base[1], global_bboxes[i][1]),
                    max(base[2], global_bboxes[i][2]), max(base[3], global_bboxes[i][3]),
                    base[4]
                ]
                base = merged_box
                global_bboxes.pop(i)
            else:
                i += 1
        merged_bboxes.append(base)
    return merged_bboxes


def process_model_output(output, img, threshold, num_classes, distance_threshold, img_dims, global_offset):
    """Process model output, apply non-max suppression, and convert to global coordinates."""
    output = util.non_max_suppression(output, conf_threshold=0.1, iou_threshold=0.1, nc=num_classes)

    bboxes = []
    for detection in output:
        detection = detection.view(-1, 6)
        for x1, y1, x2, y2, conf, class_id in detection:
            if conf.item() >= threshold:
                # Convert to global coordinates
                gx1, gy1 = x1.item() + global_offset[0], y1.item() + global_offset[1]
                gx2, gy2 = x2.item() + global_offset[0], y2.item() + global_offset[1]
                bboxes.append([gx1, gy1, gx2, gy2, int(class_id.item())])

    return bboxes


@torch.no_grad()
def run(args, data, model):
    """Main function to process images, detect objects, and output results with timing metrics."""
    out_dir = args.data_out
    os.makedirs(out_dir, exist_ok=True)

    grid_batch_size = args.batch_size * 4
    model_times, iteration_times = [], []
    start_collecting = False

    for iteration in range(args.num_iterations):
        start_iteration = time()

        for i in range(0, len(data), args.batch_size):
            batch_paths = data[i:i + args.batch_size]
            batch_processed_images = []

            for path in batch_paths:
                original_image = preprocess_image(path, args.grid_size)
                grid_images = split_image_into_grid(original_image, args.grid_size)
                rows, cols = args.grid_size
                h, w, _ = original_image.shape
                grid_h, grid_w = h // rows, w // cols
                global_bboxes = []

                for img_index, img in enumerate(grid_images):
                    # Determine global offset for current grid section
                    global_offset = ((img_index % cols) * grid_w, (img_index // cols) * grid_h)

                    x = torch.from_numpy(np.transpose(img, (2, 0, 1))).cuda().half() / 255.0
                    x = x.unsqueeze(0)  # Add batch dimension

                    start_model = time()
                    output = model(x)
                    end_model = time()
                    model_times.append(end_model - start_model)

                    img_dims = (img.shape[1], img.shape[0])

                    # Process model output and convert to global coordinates
                    bboxes= process_model_output(output, img, args.threshold, args.num_classes, 
                                                  args.distance_threshold, img_dims, global_offset)
                    global_bboxes.extend(bboxes)

                # Merge global bounding boxes
                merged_global_bboxes = merge_global_bboxes(global_bboxes, args.distance_threshold)

                total_holes, total_cracks = 0, 0

                # Define the final bbox output path for the merged bboxes
                final_bbox_output_path = os.path.join(out_dir, 'bbox_coordinates', 
                                                      os.path.splitext(os.path.basename(path))[0] + ".txt")
                os.makedirs(os.path.dirname(final_bbox_output_path), exist_ok=True)
                
                # Reassemble the image from grid sections for visualization
                reassembled_image = reassemble_image_from_grid(grid_images, original_image.shape[:2], args.grid_size)

                img_width , img_height = (reassembled_image.shape[1], reassembled_image.shape[0])
                # Draw merged bounding boxes on the reassembled image
                with open (final_bbox_output_path, 'w') as f:
                    for bbox in merged_global_bboxes:
                        x1, y1, x2, y2, class_id = map(int, bbox[:5])
                        y1 = max(y1 - 7 , 0)
                        y2 = max(y2 - 7 , 0)

                        # Calculate the center coordinates, width, and height of the bounding box
                        bx_center = (x1 + x2) / 2
                        by_center = (y1 + y2) / 2
                        bwidth = x2 - x1
                        bheight = y2 - y1

                        # Normalize the center coordinates, width, and height by the image dimensions
                        img_width, img_height = reassembled_image.shape[1], reassembled_image.shape[0]
                        nx_center = bx_center / img_width
                        ny_center = by_center / img_height
                        nwidth = bwidth / img_width
                        nheight = bheight / img_height

                        # Write the bounding box in YOLO format to the file
                        f.write(f"{class_id} {nx_center:.6f} {ny_center:.6f} {nwidth:.6f} {nheight:.6f}\n")


                        if class_id == 0:
                            total_holes += 1 
                        elif class_id == 1:
                            total_cracks += 1

                        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        cv2.rectangle(reassembled_image, (x1, y1), (x2, y2), color, 2) 
                
                # Annotate the reassembled image with total counts
                cv2.putText(reassembled_image, f'Total Holes: {total_holes}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(reassembled_image, f'Total Cracks: {total_cracks}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                batch_processed_images.append(reassembled_image)

            # Write processed images to disk
            for img, path in zip(batch_processed_images, batch_paths):
                output_path = os.path.join(out_dir, os.path.basename(path))
                cv2.imwrite(output_path, img)

        end_iteration = time()
        iteration_times.append(end_iteration - start_iteration)

        if iteration == 0:
            start_collecting = True

    # Calculate and log average model and iteration times
    average_model_time = sum(model_times) / len(model_times) if model_times else 0
    average_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0

    print(f'Average MODEL execution time: {average_model_time:.4f} seconds')
    print(f'Average Inference time over {args.num_iterations} iterations: {average_iteration_time:.4f} seconds')
    print(f'Batch Processing: {grid_batch_size} Grid Images per Batch')

# #gather data from multiple directories
# def get_data(root_dir, depth=3):
#     dataset_loader = []
#     def gather_images(dir_path, current_depth):
#         if current_depth > depth:
#             return
#         for item in sorted(os.listdir(dir_path)):
#             item_path = os.path.join(dir_path, item)
#             if os.path.isdir(item_path):
#                 # Recurse into the directory if the depth limit hasn't been reached
#                 gather_images(item_path, current_depth + 1)
#             elif item.endswith((".png", ".jpg", ".tiff")):
#                 dataset_loader.append(item_path)

#     # Start gathering images from the root directory
#     gather_images(root_dir, current_depth=1)
#     return dataset_loader

def get_data(data_root):
    dataset_loader = []
    for file in sorted(os.listdir(data_root)):
        if file.endswith((".png",".jpg","tiff")):
            dataset_loader.append(os.path.join(data_root, file))
    return dataset_loader


def load_model(model_path):
    """Load the PyTorch model from a specified path."""
    model = torch.load(model_path, map_location='cuda')['model'].float().fuse()
    model.half()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Image processing and object detection script.")
    parser.add_argument('--batch-size', default=7, type=int, help="Number of images to process in a batch.")
    parser.add_argument('--num-classes', default=2, type=int, help="Number of classes for detection.")
    parser.add_argument('--threshold', default=0.1, type=float, help="Confidence threshold for detections.")
    parser.add_argument('--num_iterations', default=2, type=int, help="Number of iterations to run.")
    parser.add_argument('--distance-threshold', default=50, type=int, help="Distance threshold for merging close boxes.")
    parser.add_argument('--grid-size', nargs=2, default=[2, 2], type=int, help="Grid size to split images into")
    parser.add_argument('--model-path', default='weights/v1v6/s/1280Grid.pt', type=str)
    parser.add_argument('--data-root', default='data/', type=str)
    parser.add_argument('--data-out', default='data/saves', type=str)
    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()

    data = get_data(args.data_root)
    model = load_model(args.model_path)
    run(args, data, model)

if __name__ == "__main__":
    main()
