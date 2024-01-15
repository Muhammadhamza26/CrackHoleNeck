import math
import random
import threading
from time import sleep, time

import copy
import cv2
import numpy
import torch
import torchvision
from torch.nn.functional import cross_entropy


def display(name, img):
    cv2.namedWindow(name, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)

    from torch.backends import cudnn

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != "Windows":
        torch.multiprocessing.set_start_method("fork", force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if "OMP_NUM_THREADS" not in environ:
        environ["OMP_NUM_THREADS"] = "1"

    # setup MKL threads
    if "MKL_NUM_THREADS" not in environ:
        environ["MKL_NUM_THREADS"] = "1"

def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float()
    x = torch.zeros((1, 3, args.input_size, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      './weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')
    # Inference example
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py

def wh2xy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale(output, image, x):
    shape = image.shape
    # Scale outputs
    gain = min(x.shape[2] / shape[0], x.shape[3] /
               shape[1])  # gain  = old / new
    pad = (x.shape[3] - shape[1] * gain) / 2, (
        x.shape[2] - shape[0] * gain
    ) / 2  # wh padding

    output[:, [0, 2]] -= pad[0]  # x padding
    output[:, [1, 3]] -= pad[1]  # y padding
    output[:, :4] /= gain

    output[:, 0].clamp_(0, shape[1])  # x1
    output[:, 1].clamp_(0, shape[0])  # y1
    output[:, 2].clamp_(0, shape[1])  # x2
    output[:, 3].clamp_(0, shape[0])  # y2
    return output

def make_anchors(x, strides, offset=0.5):
    anchors, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchors), torch.cat(stride_tensor)


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def resize(image, input_size):
    shape = image.shape[:2]  # current shape [height, width]
    r = min(1.0, input_size / shape[0], input_size / shape[1])
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))

    w = numpy.mod(input_size - pad[0], 32) / 2
    h = numpy.mod(input_size - pad[1], 32) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT)

    # Convert BGR to GRAY
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = numpy.stack((image, image, image), axis=-1) ### original line
    ###################################################################
    if len(image.shape) == 2:  # Check if the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #################################################################
    return image


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT)

    # Convert BGR to GRAY
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = numpy.stack((image, image, image), axis=-1)
    return image


def non_max_suppression(outputs, conf_threshold, iou_threshold, nc):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = nc or (outputs.shape[1] - 4)  # number of classes
    nm = outputs.shape[1] - nc - 4
    # outputs.shape[1] is 6
    mi = 4 + nc  # mask start index
    xc = outputs[:, 4:mi].amax(1) > conf_threshold  # candidates
    # Settings
    start = time()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    nms_outputs = [torch.zeros((0, 6 + nm), device=outputs.device)] * bs
    for index, output in enumerate(outputs):  # image index, image inference
        output = output.transpose(0, -1)[xc[index]]  # confidence
        # If none remain process next image
        if not output.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = output.split((4, nc, nm), 1)
        # mask is zero since the nm is zero

        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        # if nc > 1:
        i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
        output = torch.cat(
            (box[i], output[i, 4 + j, None], j[:, None].float(), mask[i]), 1
        )
        # print(
        #     f"box[i]: {box[i]} \noutput: {output[i, 4+j, None]} \nj: {j[:, None].float()} \nmask: {mask[i]} "
        # )
        # else:  # best class only
        #     conf, j = cls.max(1, keepdim=True)
        #     output = torch.cat((box, conf, j.float(), mask), 1)[
        #         conf.view(-1) > conf_threshold
        #     ]
        # print(f"output: {output}")
        # Check shape
        n = output.shape[0]  # number of boxes
        # print(f"Number of boxes: {n}")
        if not n:  # no boxes
            continue
        output = output[
            output[:, 4].argsort(descending=True)[:max_nms]
        ]  # sort by confidence and remove excess boxes
        # print(f"sorted output based on the conf: {output}")
        # Batched NMS
        c = output[:, 5:6] * max_wh  # classes
        # print(f"c: {c}")
        boxes, scores = (
            output[:, :4] + c,
            output[:, 4],
        )  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections

        nms_outputs[index] = output[i]
        if (time() - start) > time_limit:
            break  # time limit exceeded

    return nms_outputs

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


class Cameras1:
    def __init__(self, sources, input_size=640):
        self.input_size = input_size

        n = len(sources)
        self.sources = sources
        self.images, self.fps, self.frames, self.threads = (
            [None] * n,
            [0] * n,
            [0] * n,
            [None] * n,
        )

        for i, source in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {source}... "
            camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            camera.set(cv2.CAP_PROP_FOURCC, fourcc)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            assert camera.isOpened(), f"{st}Failed to open {source}"
            w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camera.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(camera.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )
            self.fps[i] = (
                max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            )  # 30 FPS fallback
            print(fps)
            ret, self.images[i] = camera.read()  # guarantee first frame
            if ret:
                self.threads[i] = threading.Thread(
                    target=self.update, args=([i, camera, source]), daemon=True
                )
                print(
                    f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)"
                )
                self.threads[i].start()
            else:
                print("Error reading frame")
        print()  # newline

    def update(self, i, camera, stream):
        # Read stream `i` frames in daemon thread
        while camera.isOpened():
            camera.grab()  # .read() = .grab() followed by .retrieve()
            success, image = camera.retrieve()
            if success:
                self.images[i] = image
            else:
                print(
                    "WARNING ⚠️ Video stream unresponsive, please check your IP camera connection."
                )
                self.images[i] = numpy.zeros_like(self.images[i])
                camera.open(stream)  # re-open stream if signal was lost
            sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration

        images = self.images.copy()

        x = numpy.stack([resize(i, self.input_size) for i in images], axis=0)
        x = x.transpose((0, 3, 1, 2))  # HWC to CHW
        x = numpy.ascontiguousarray(x)  # contiguous

        return self.sources, images, x

    def __len__(self):
        return len(self.sources)


class Cameras:
    def __init__(self, sources, input_size=640):
        # ... (existing code)
        self.input_size = input_size

        self.processed_images = [None] * \
            len(sources)  # Buffer for processed images
        self.lock = threading.Lock()  # Lock for synchronizing access

        n = len(sources)
        self.sources = sources
        self.images, self.fps, self.frames, self.threads = (
            [None] * n,
            [0] * n,
            [0] * n,
            [None] * n,
        )
        for i, source in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {source}... "
            camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            camera.set(cv2.CAP_PROP_FOURCC, fourcc)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            assert camera.isOpened(), f"{st}Failed to open {source}"
            w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camera.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(camera.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )
            self.fps[i] = (
                max((fps if math.isfinite(fps) else 0) % 100, 0) or 30
            )  # 30 FPS fallback

            ret, self.images[i] = camera.read()  # guarantee first frame
            if ret:
                self.threads[i] = threading.Thread(
                    target=self.update, args=([i, camera, source]), daemon=True
                )
                print(
                    f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)"
                )
                self.threads[i].start()
            else:
                print("Error reading frame")
        print()  # newline

    def update(self, i, camera, stream):

        while camera.isOpened():
            camera.grab()  # .read() = .grab() followed by .retrieve()
            success, image = camera.retrieve()
            if success:
                with self.lock:  # Acquire lock before modifying shared resources
                    self.images[i] = image
                    self.processed_images[i] = None  # Clear processed image
            else:
                print(
                    "WARNING ⚠️ Video stream unresponsive, please check your IP camera connection."
                )
                self.images[i] = numpy.zeros_like(self.images[i])
                camera.open(stream)  # re-open stream if signal was lost
            sleep(0.0)  # wait time

    # New method to set processed image
    def set_processed_image(self, i, processed_image):
        with self.lock:  # Acquire lock before modifying shared resources
            self.processed_images[i] = processed_image

    # New method to get processed image
    def get_processed_image(self, i):
        with self.lock:  # Acquire lock before accessing shared resources
            return self.processed_images[i]

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration

        images = self.images.copy()

        x = numpy.stack([resize(i, self.input_size) for i in images], axis=0)
        x = x.transpose((0, 3, 1, 2))  # HWC to CHW
        x = numpy.ascontiguousarray(x)  # contiguous

        return self.sources, images, x

    def __len__(self):
        return len(self.sources)
    
    
def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def compute_iou(box1, box2, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU

def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(ckpt, model):
    dst = model.state_dict()
    src = torch.load(ckpt, 'cpu')['model'].float().state_dict()
    ckpt = {}
    for k, v in src.items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def weight_decay(model, decay):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class Assigner(torch.nn.Module):
    """
    Task-aligned One-stage Object Detection assigner
    """

    def __init__(self, top_k=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        size = pd_scores.size(0)
        max_boxes = gt_bboxes.size(1)

        if max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))
        # get in_gts mask, (b, max_num_obj, h*w)
        n_anchors = anc_points.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        bbox_deltas = bbox_deltas.view(bs, n_boxes, n_anchors, -1)
        mask_in_gts = bbox_deltas.amin(3).gt_(1e-9)
        # get anchor_align metric, (b, max_num_obj, h*w)
        na = pd_bboxes.shape[-2]
        true_mask = (mask_in_gts * mask_gt).bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([size, max_boxes, na],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([size, max_boxes, na],
                                  dtype=pd_scores.dtype, device=pd_scores.device)
        index = torch.zeros([2, size, max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        index[0] = torch.arange(end=size).view(-1, 1).repeat(1, max_boxes)  # b, max_num_obj
        index[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores[true_mask] = pd_scores[index[0], :, index[1]][true_mask]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).repeat(1, max_boxes, 1, 1)[true_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).repeat(1, 1, na, 1)[true_mask]
        overlaps[true_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        # get top_k_metric mask, (b, max_num_obj, h*w)
        num_anchors = align_metric.shape[-1]  # h*w
        top_k_mask = mask_gt.repeat([1, 1, self.top_k]).bool()
        # (b, max_num_obj, top_k)
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        # (b, max_num_obj, top_k)
        top_k_indices.masked_fill_(~top_k_mask, 0)
        # (b, max_num_obj, top_k, h*w) -> (b, max_num_obj, h*w)
        count = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k):
            count.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        # filter invalid bboxes
        count.masked_fill_(count > 1, 0)
        mask_top_k = count.to(align_metric.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * mask_gt
        # (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, max_boxes, 1])  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target, assigned target labels, (b, 1)
        batch_index = torch.arange(end=size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_idx = target_gt_idx + batch_index * max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
        target_scores = target_scores * (norm_align_metric.amax(-2).unsqueeze(-1))

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


class BoxLoss(torch.nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self.df_loss(pred_dist[fg_mask].view(-1, self.dfl_ch + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    def df_loss(pred_dist, target):
        # Distribution Focal Loss (DFL)
        # https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        left_loss = cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class ComputeLoss:
    def __init__(self, model, params):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.no = m.no
        self.nc = m.nc  # number of classes
        self.dfl_ch = m.ch
        self.params = params
        self.device = device
        self.stride = m.stride  # model strides

        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(top_k=10, num_classes=self.nc, alpha=0.5, beta=6.0)

        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    def __call__(self, outputs, targets):
        shape = outputs[0].shape
        loss = torch.zeros(3, device=self.device)  # cls, box, dfl

        x_cat = torch.cat([i.view(shape[0], self.no, -1) for i in outputs], 2)
        pred_distri, pred_scores = x_cat.split((self.dfl_ch * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        size = torch.tensor(shape[2:], device=self.device, dtype=pred_scores.dtype)
        size = size * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(outputs, self.stride, 0.5)

        # targets
        indices = targets['idx'].view(-1, 1)
        batch_size = pred_scores.shape[0]
        box_targets = torch.cat((indices, targets['cls'].view(-1, 1), targets['box']), 1)
        box_targets = box_targets.to(self.device)
        if box_targets.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = box_targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = box_targets[matches, 1:]
            x = gt[..., 1:5].mul_(size[[1, 0, 1, 0]])
            y = x.clone()
            y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
            y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
            y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
            y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
            gt[..., 1:5] = y
        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.box_decode(anchor_points, pred_distri, self.project)  # xyxy, (b, h*w, 4)

        assigned_targets = self.assigner(pred_scores.detach().sigmoid(),
                                         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                         anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask, target_gt_idx = assigned_targets

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        loss[0] = self.cls_loss(pred_scores, target_scores.to(pred_scores.dtype)).sum()  # BCE
        loss[0] = loss[0] / target_scores_sum

        if fg_mask.sum():
            # box loss
            target_bboxes /= stride_tensor
            loss[1], loss[2] = self.box_loss(pred_distri,
                                             pred_bboxes,
                                             anchor_points,
                                             target_bboxes,
                                             target_scores,
                                             target_scores_sum, fg_mask)

        loss[0] *= self.params['cls']  # cls gain
        loss[1] *= self.params['box']  # box gain
        loss[2] *= self.params['dfl']  # dfl gain

        return loss.sum()

    @staticmethod
    def box_decode(anchor_points, pred_dist, project):
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(project.type(pred_dist.dtype))
        a, b = pred_dist.chunk(2, -1)
        a = anchor_points - a
        b = anchor_points + b
        return torch.cat((a, b), -1)
