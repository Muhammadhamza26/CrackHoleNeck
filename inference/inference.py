import argparse
import warnings

import cv2
import torch

from utils import util

warnings.filterwarnings("ignore")


@torch.no_grad()
def run(args):
    model = torch.load('./weights/best_v8s_newsetup.pt', 'cuda')['model'].float().fuse()
    # model = torch.load('./weights/v8_s.pt', 'cpu')['model'].float().fuse()

    model.half()
    model.eval()

    buffer_size = 65536

    # stream1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # stream2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # stream1.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    # stream2.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

    cameras = [2, 4, 6, 8, 10, 12]
    cameras = util.Cameras(cameras, args.input_size)
    pts = {}
    for sources, images, x in cameras:
        x = torch.from_numpy(x).cuda()
        x = x.half()  # uint8 to fp16/32
        x = x / 255.  # 0 - 255 to 0.0 - 1.0
        # Inference
        outputs = model(x)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.1, 0.5, model.head.nc)
        for i, (source, output, image) in enumerate(zip(sources, outputs, images)):
            shape = image.shape
            centers = []
            # Scale outputs
            gain = min(x.shape[2] / shape[0], x.shape[3] /
                       shape[1])  # gain  = old / new
            pad = (x.shape[3] - shape[1] * gain) / \
                2, (x.shape[2] - shape[0] * gain) / 2  # wh padding

            output[:, [0, 2]] -= pad[0]  # x padding
            output[:, [1, 3]] -= pad[1]  # y padding
            output[:, :4] /= gain

            output[:, 0].clamp_(0, shape[1])  # x1
            output[:, 1].clamp_(0, shape[0])  # y1
            output[:, 2].clamp_(0, shape[1])  # x2
            output[:, 3].clamp_(0, shape[0])  # y2
            # Draw boxes
            num_holes = 0
            for box in output:
                x1, y1, x2, y2 = list(map(int, box[:4]))
                center = [(x1+x2)/2, (y1+y2)/2]
                if int(box.cpu().numpy()[5]) == 0:
                    num_holes += 1
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                centers.append(center)
            pts[source] = centers
            cv2.putText(image,
                        f'Number of Holes: {num_holes}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2, cv2.LINE_AA)
            pts[source] = centers
            h, w = image.shape[:2]
            h, w = 3 * h // 4, 3 * w // 4
            image = cv2.resize(image, (w, h))
            cv2.imshow(winname=str(source), mat=image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=1280, type=int)
    args = parser.parse_args()

    util.setup_seed()
    util.setup_multi_processes()

    run(args)


if __name__ == "__main__":
    main()
