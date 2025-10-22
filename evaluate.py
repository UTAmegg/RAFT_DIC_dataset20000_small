import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import core.datasets_v0 as datasets
from core.utils import flow_viz
from core.utils import frame_utils

from core.raft import RAFT
from core.utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def validate_dic(model, args, iters=12):
    """ Validate on DIC dataset """
    model.eval()

    # Create test dataset
    test_dataset = datasets.DICDataset(aug_params=None, split='test', root=args.dic_root)

    if len(test_dataset) == 0:
        print("Error: No test data found in", args.dic_root)
        return {'dic-epe': 0.0}

    epe_list = []
    px1_list = []
    px3_list = []
    px5_list = []

    print(f"Evaluating {len(test_dataset)} DIC image pairs...")

    for val_id in range(len(test_dataset)):
        img1, img2, flow_gt, valid = test_dataset[val_id]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()

        # Run model
        flow_predictions = model(img1, img2, iters=iters, test_mode=True)

        if isinstance(flow_predictions, tuple):
            _, flow_pr = flow_predictions  # Get both low and high res flow
        else:
            flow_pr = flow_predictions[-1]  # Get last prediction

        # Calculate EPE
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        # Calculate pixel accuracy metrics
        epe_flat = epe.view(-1).numpy()
        px1_list.append(np.mean(epe_flat < 1))
        px3_list.append(np.mean(epe_flat < 3))
        px5_list.append(np.mean(epe_flat < 5))

        if val_id % 10 == 0:
            print(f"Processed {val_id + 1}/{len(test_dataset)} images")

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(px1_list)
    px3 = np.mean(px3_list)
    px5 = np.mean(px5_list)

    print("\n" + "=" * 50)
    print("DIC Validation Results:")
    print("=" * 50)
    print("EPE: %.4f" % epe)
    print("1px: %.4f (%.2f%%)" % (px1, px1 * 100))
    print("3px: %.4f (%.2f%%)" % (px3, px3 * 100))
    print("5px: %.4f (%.2f%%)" % (px5, px5 * 100))
    print("=" * 50)

    return {'dic-epe': epe, 'dic-1px': px1, 'dic-3px': px3, 'dic-5px': px5}


@torch.no_grad()
def create_dic_submission(model, args, iters=12, output_path='dic_submission'):
    """ Create displacement predictions for DIC dataset """
    model.eval()
    test_dataset = datasets.DICDataset(split='test', aug_params=None, root=args.dic_root)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Creating DIC predictions for {len(test_dataset)} image pairs...")

    for test_id in range(len(test_dataset)):
        image1, image2, img_id = test_dataset[test_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        # Run model
        flow_predictions = model(image1, image2, iters=iters, test_mode=True)

        if isinstance(flow_predictions, tuple):
            _, flow_pr = flow_predictions
        else:
            flow_pr = flow_predictions[-1]

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()

        # Save as .flo file
        output_file = os.path.join(output_path, f'{img_id}_flow.flo')
        frame_utils.writeFlow(output_file, flow)

        # Optionally save as separate u and v numpy files
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        np.save(os.path.join(output_path, f'{img_id}_u.npy'), u)
        np.save(os.path.join(output_path, f'{img_id}_v.npy'), v)

        # Save visualization
        flow_img = flow_viz.flow_to_image(flow)
        Image.fromarray(flow_img).save(os.path.join(output_path, f'{img_id}_flow_viz.png'))

        if test_id % 10 == 0:
            print(f"Processed {test_id + 1}/{len(test_dataset)} images")

    print(f"Results saved to {output_path}")


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the KITTI leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Perform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Perform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

    # DIC specific arguments
    parser.add_argument('--dic_root', default='G:/20251017_RAFTcorr_training_dataset_20000',
                        help='root directory for DIC dataset')
    parser.add_argument('--save_predictions', action='store_true', help='save predicted flow fields')
    parser.add_argument('--output_path', default='dic_predictions', help='output path for predictions')

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'dic':
            if args.save_predictions:
                create_dic_submission(model.module, args, output_path=args.output_path)
            else:
                validate_dic(model.module, args)

        elif args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)