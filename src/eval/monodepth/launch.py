import torch
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from collections import defaultdict
from eval.monodepth.metadata import dataset_metadata
from add_ckpt_path import add_path_to_dust3r


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights", type=str, help="path to the model weights", default=""
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument("--output_dir", type=str, default="", help="value for outdir")
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )
    parser.add_argument(
        "--full_seq", type=bool, default=False, help="whether to use all seqs"
    )
    parser.add_argument("--seq_list", default=None)

    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    parser.add_argument("--eviction", action="store_true")
    parser.add_argument("--P", type=float, default=0.8)
    parser.add_argument("--temp", type=float, default=0.5)
    return parser


def eval_mono_depth_estimation(args, model, device):
    metadata = dataset_metadata.get(args.eval_dataset)
    if metadata is None:
        raise ValueError(f"Unknown dataset: {args.eval_dataset}")

    img_path = metadata.get("img_path")
    if "img_path_func" in metadata:
        img_path = metadata["img_path_func"](args)

    process_func = metadata.get("process_func")
    if process_func is None:
        raise ValueError(
            f"No processing function defined for dataset: {args.eval_dataset}"
        )

    for filelist, save_dir in process_func(args, img_path):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        eval_mono_depth(args, model, device, filelist, save_dir=save_dir)


def eval_mono_depth(args, model, device, filelist, save_dir=None):
    model.eval()
    load_img_size = 518
    for file in tqdm(filelist):
        # construct the "image pair" for the single image
        file = [file]
        images = load_images(
            file, size=load_img_size, verbose=False, crop=not args.no_crop
        )
        views = []
        num_views = len(images)

        for i in range(num_views):
            view = {
                "img": images[i]["img"].to(device='cuda'),
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ).to(device='cuda'),
                "true_shape": torch.from_numpy(images[i]["true_shape"]).to(device='cuda'),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(
                    0
                ).to(device='cuda'),
                "img_mask": torch.tensor(True).unsqueeze(0).to(device='cuda'),
                "ray_mask": torch.tensor(False).unsqueeze(0).to(device='cuda'),
                "update": torch.tensor(True).unsqueeze(0).to(device='cuda'),
                "reset": torch.tensor(False).unsqueeze(0).to(device='cuda'),
            }
            views.append(view)
        for view in views:
            view['img'] = (view['img'] + 1.0) / 2.0
        outputs = loss_of_one_batch(views, model, device, None, inference=True, eviction=args.eviction, P=args.P, temp=args.temp)
        pts3ds_self = [output["depth"].cpu() for output in outputs["pred"]]
        depth_map = pts3ds_self[0][..., -1].mean(dim=0)

        if save_dir is not None:
            # save the depth map to the save_dir as npy
            np.save(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.npy')}",
                depth_map.cpu().detach().numpy(),
            )
            # also save the png
            depth_map = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )
            depth_map = (depth_map * 255).cpu().detach().numpy().astype(np.uint8)
            cv2.imwrite(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.png')}",
                depth_map,
            )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    print('args: ', args)
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.inference import loss_of_one_batch
    from dust3r.model import ARCroco3DStereo

    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    from streamvggt.utils.geometry import unproject_depth_map_to_point_map
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy
    model = StreamVGGT()
    ckpt = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to("cuda")
    del ckpt
    eval_mono_depth_estimation(args, model, args.device)
