import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params["lrf"]) + params["lrf"]

    return fn


def train(args, params):
    util.setup_seed()
    util.setup_multi_processes()

    # Model
    model = nn.yolo_v8_s(len(params["names"]))  # arg --> #classes
    model = util.load_weight("./weights/v8_s.pt", model)
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params["weight_decay"] *= args.batch_size * \
        args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(
        util.weight_decay(model, params["weight_decay"]),
        params["lr0"],
        params["momentum"],
        nesterov=True,
    )

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    train_path = "./Dataset/images/train"
    filenames = [
        f"{train_path}/{filename}" for filename in sorted(os.listdir(train_path))
    ]

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(
        dataset,
        args.batch_size,
        sampler is None,
        sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[
                args.local_rank], output_device=args.local_rank
        )

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params["warmup_epochs"] * num_batch), 100)
    with open("weights/step.csv", "a") as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "loss", "Recall",
                               "Precision", "mAP@50", "mAP"]
            )
            writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(("\n" + "%10s" * 3) % ("epoch", "memory", "loss"))
            if args.local_rank == 0:
                p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

            optimizer.zero_grad()
            m_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params["warmup_bias_lr"],
                                  y["initial_lr"] * lr(epoch)]
                        else:
                            fp = [0.0, y["initial_lr"] * lr(epoch)]
                        y["lr"] = numpy.interp(x, xp, fp)
                        if "momentum" in y:
                            fp = [params["warmup_momentum"], params["momentum"]]
                            y["momentum"] = numpy.interp(x, xp, fp)

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)  # forward
                loss = criterion(outputs, targets)

                m_loss.update(loss.item(), samples.size(0))

                loss *= args.batch_size  # loss scaled by batch_size
                loss *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if args.local_rank == 0:
                    # (GB)
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"
                    s = ("%10s" * 2 + "%10.4g") % (
                        f"{epoch + 1}/{args.epochs}",
                        memory,
                        m_loss.avg,
                    )
                    p_bar.set_description(s)

                del loss
                del outputs

            # Scheduler
            scheduler.step()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)
                writer.writerow(
                    {
                        "epoch": str(epoch + 1).zfill(3),
                        "loss": str(f"{m_loss.avg:.3f}"),
                        "mAP": str(f"{last[0]:.3f}"),
                        "mAP@50": str(f"{last[1]:.3f}"),
                        "Recall": str(f"{last[2]:.3f}"),
                        "Precision": str(f"{last[3]:.3f}"),
                    }
                )
                f.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                ckpt = {"model": copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(ckpt, "./weights/last.pt")
                if best == last[0]:
                    torch.save(ckpt, "./weights/best.pt")
                del ckpt

    if args.local_rank == 0:
        util.strip_optimizer("./weights/best.pt")  # strip optimizers
        util.strip_optimizer("./weights/last.pt")  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    val_path = "./Dataset/images/val"
    filenames = [
        f"{val_path}/{filename}" for filename in sorted(os.listdir(val_path))]
    numpy.random.shuffle(filenames)
    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(
        dataset,
        args.batch_size,
        False,
        num_workers=8,
        pin_memory=True,
        collate_fn=Dataset.collate_fn,
    )

    if model is None:
        model = torch.load("./weights/best.pt",
                           map_location="cuda")["model"].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.0
    m_rec = 0.0
    map50 = 0.0
    mean_ap = 0.0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=("%10s" * 3) %
                      ("precision", "recall", "mAP"))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.head.nc)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets["idx"] == i
            cls = targets["cls"][idx]
            box = targets["box"][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(
                output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append(
                        (metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1))
                    )
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat((cls, util.wh2xy(box) * scale), 1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append(
                (metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy()
               for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    # Print results
    print("%10.3g" * 3 % (m_pre, m_rec, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    from thop import profile, clever_format

    model = nn.yolo_v8_s(len(params["names"]))
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    macs, params = profile(model, inputs=(torch.zeros(shape),), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    if args.local_rank == 0:
        print(f"MACs: {macs}")
        print(f"Parameters: {params}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-size", default=1280, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--train", default=True, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")

    args = parser.parse_args()

    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    args.distributed = int(os.getenv("WORLD_SIZE", 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")

    if args.local_rank == 0 and not os.path.exists("weights"):
        os.makedirs("weights")

    with open("utils/args.yaml", errors="ignore") as f:
        params = yaml.safe_load(f)
    profile(args, params)
    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()
