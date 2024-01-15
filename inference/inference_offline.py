import os
import cv2
import torch
from tqdm import tqdm

import argparse
import warnings
import numpy as np
from utils import util
from time import time
import subprocess

warnings.filterwarnings("ignore")

def get_gpu_memory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    # Convert byte string to string, split by line, and convert to integer
    return [int(x) for x in result.decode('utf-8').strip().split('\n')]

# Before running the inference code
memory_before = get_gpu_memory()

OUT_DIR = "/hdd/KIA_Smart_Factory_Crack_Hole_detection/hamza/Infer/15Nov_40cm/images/runs/1080X_corrected/new"
os.makedirs(OUT_DIR, exist_ok= True)

def get_data():
    ROOT = "/hdd/KIA_Smart_Factory_Crack_Hole_detection/hamza/Infer/15Nov_40cm/images/28_1280"
    dataset_loader = []
    for file in os.listdir(ROOT):
        if file.endswith(".png"):
            file_path = os.path.join(ROOT, file)
            dataset_loader.append(file_path)
    return dataset_loader

@torch.no_grad()
def run(args, data):
    model = torch.load('./weights/allData/X/1280(dataset0+parts_Val,v0-4_Train)inverted.pt', 'cuda')['model'].float().fuse()
    model.half()
    model.eval()

    batch_size = 28
    num_iterations = 1
    total_model_time = 0
    total_iteration_time = 0
    iteration_times = []
    start_collecting = False 

    for iteration in range(num_iterations):
        start_iteration = time()

        # Process a batch of images
        for i in range(0, len(data), batch_size):
            batch_paths = data[i:i + batch_size]
            batch_images = [cv2.imread(path) for path in batch_paths]
            batch_images = [util.resize(image, args.input_size) for image in batch_images]
            batch_images = [np.transpose(image, (2, 0, 1)) for image in batch_images]  # HWC to CHW
            batch_images = np.stack(batch_images, axis=0)
            x = torch.from_numpy(batch_images).cuda().half() / 255.0

            # Inference
            start_model = time()
            output = model(x)
            elapsed_model = time() - start_model
            
            if start_collecting:
                total_model_time += elapsed_model
                print(f'Batch {i // batch_size + 1}, Model Execution time: {elapsed_model} seconds')

            # NMS
            output = util.non_max_suppression(output, 0.1, 0.5, model.head.nc)

            # Process each image in the batch
            for j, single_output in enumerate(output):
                path = batch_paths[j]
                filename = path.split('/')[-1]
                image = cv2.imread(path)
                shape = image.shape
                # print("Shape of original image:", shape)

                gain = min(x[j].shape[1] / shape[0], x[j].shape[2] / shape[1])  # gain = old / new
                pad = (x[j].shape[2] - shape[1] * gain) / 2, (x[j].shape[1] - shape[0] * gain) / 2  # wh padding


                if single_output is None or len(single_output) == 0:
                    # No detections for this image
                    continue

                if single_output.ndim == 1:
                    # If single_output is 1D, reshape it to 2D (1, -1)
                    single_output = single_output.view(1, -1)

                single_output[:, [0, 2]] -= pad[0]  # x padding
                single_output[:, [1, 3]] -= pad[1]  # y padding
                single_output[:, :4] /= gain

                single_output[:, 0].clamp_(0, shape[1])  # x1
                single_output[:, 1].clamp_(0, shape[0])  # y1
                single_output[:, 2].clamp_(0, shape[1])  # x2
                single_output[:, 3].clamp_(0, shape[0])  # y2

                # Draw boxes
                num_holes = 0
                num_cracks = 0
                for box in single_output:
                    x1, y1, x2, y2 = list(map(int, box[:4]))
                    x_center, y_center = [(x1+x2)//2, (y1+y2)//2]
                    radius = max(x2 - x1, y2 - y1) // 2
                    if int(box.cpu().numpy()[5]) == 0:
                        num_holes += 1
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.circle(image, (x_center, y_center), radius, (0, 255, 0), 2)
                    else:
                        num_cracks +=1
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

                cv2.putText(image, f'Number of Holes: {num_holes}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Number of Cracks: {num_cracks}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2, cv2.LINE_AA)
                h, w = image.shape[:2]
                h, w = 3 * h // 4, 3 * w // 4
                image = cv2.resize(image, (w, h))
                cv2.imwrite(os.path.join(OUT_DIR, filename), image)

        elapsed_iteration = time() - start_iteration
        if start_collecting:
            iteration_times.append(elapsed_iteration)
            total_iteration_time += elapsed_iteration
            print(f'Iteration {iteration + 1}, Execution time: {elapsed_iteration} seconds')

        if iteration == 0:
            start_collecting = True
    average_model_time = total_model_time / (num_iterations * len(data) / batch_size)
    average_iteration_time = total_iteration_time / num_iterations
    print(f'Average MODEL execution time: {average_model_time} seconds')
    print(f'Average Inference time over {num_iterations} iterations: {average_iteration_time} seconds')
    # print(f'Individual iteration times: {iteration_times}')
    print(f'Batch Processing: {batch_size} Images per Batch')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=1280, type=int)
    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()
    data = get_data()
    run(args, data)

if __name__ == "__main__":
    main()
