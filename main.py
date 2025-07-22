from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from dataset import PhoenixVideoTextDataset
from cam_runner import CAMRunner, overlay_heatmap
from models.slowfast_sign import SlowFastSign
from models.twostream_slr import TwoStreamSLR


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--seq", help="sequence id or comma-separated list")
    p.add_argument("--split", default="dev", help="dataset split (e.g., train, dev, test)")
    p.add_argument(
        "--models",
        default="slowfast",
        nargs="*",
        choices=["slowfast", "twostream"],
        help="which model(s) to run",
    )
    p.add_argument("--slowfast-ckpt", help="path to SlowFast checkpoint")
    p.add_argument("--twostream-ckpt", help="path to TwoStream checkpoint")
    p.add_argument(
        "--target-from-gt",
        action="store_true",
        help="use ground-truth gloss in the middle frame as CAM target",
    )
    p.add_argument("--device", default="cuda:0", help="compute device")
    return p.parse_args()


def load_model(name: str, ckpt: str | None):
    """
    Instantiate and (optionally) load a checkpoint into the model.
    Supports 'slowfast' and 'twostream'.
    """
    if name == "slowfast":
        model = SlowFastSign()
    else:
        model = TwoStreamSLR()

    # load weights if checkpoint exists
    if ckpt is not None and Path(ckpt).exists():
        sd = torch.load(ckpt, map_location="cpu")
        # unwrap state_dict if necessary
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)

    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    # set up paths relative to this script
    repo = Path(__file__).resolve().parent
    root = repo.parent / "data" / "phoenix-2014-multisigner"
    out_root = repo.parent / "outputs"

    # build sequence list if provided
    if args.seq:
        seq_list = [s.strip() for s in args.seq.split(",")]
    else:
        seq_list = None

    # load dataset (will index by sequence IDs)
    dataset = PhoenixVideoTextDataset(root, args.split, seq_ids=seq_list)

    # iterate over all sequences in the dataset
    for i in range(len(dataset)):
        batch = dataset[i]
        seq_id = dataset.seq_ids[i]
        outputs_dir = out_root / seq_id
        outputs_dir.mkdir(parents=True, exist_ok=True)

        preds: dict[str, str] = {}
        for name in args.models:
            # choose checkpoint based on model name
            ckpt = args.slowfast_ckpt if name == "slowfast" else args.twostream_ckpt
            model = load_model(name, ckpt).to(device)

            # prepare input tensor(s)
            # if name == "slowfast":
            #     inp = batch["rgb"]
            # else:
            #     inp = [batch["rgb"], batch["pose"]]

            if name == "slowfast":
                # (T,C,H,W) ➜ (1,C,T,H,W)
                rgb5d = batch["rgb"].permute(1, 0, 2, 3).unsqueeze(0).to(device)
                inp = rgb5d
            else:
                rgb5d  = batch["rgb"].permute(1, 0, 2, 3).unsqueeze(0).to(device)
                pose5d = (
                    batch["pose"].permute(1, 0, 2, 3).unsqueeze(0).to(device)
                    if batch["pose"] is not None else None
                )
                inp = [rgb5d, pose5d]

            # compute Grad-CAMs for specified target layer(s)
            runner = CAMRunner(model, model.target_layers)
            target = None
            if args.target_from_gt and len(batch["gloss"]) > 0:
                mid = len(batch["gloss"]) // 2
                target = batch["gloss"][mid].item()
            cams = runner.run(inp, device, target)

            # save heatmap overlays per frame and per layer
            # for idx, cam in enumerate(cams):
            #     layer_dir = outputs_dir / name / f"layer{idx + 1}"
            #     layer_dir.mkdir(parents=True, exist_ok=True)
            #     for t in range(cam.shape[0]):
            #         frame = batch["rgb"][t].permute(1, 2, 0).cpu().numpy()
            #         overlay = overlay_heatmap(frame, cam[t])
            #         cv2.imwrite(str(layer_dir / f"frame_{t:03d}.png"), overlay)

            for idx, cam in enumerate(cams):
                layer_dir = outputs_dir / name / f"layer{idx + 1}"
                layer_dir.mkdir(parents=True, exist_ok=True)

                # 2-D CAM → overlay per frame
                if cam.ndim == 3:                     # (T,H,W)
                    for t in range(cam.shape[0]):
                        frame = batch["rgb"][t].permute(1, 2, 0).cpu().numpy()
                        overlay = overlay_heatmap(frame, cam[t])
                        cv2.imwrite(str(layer_dir / f"frame_{t:03d}.png"), overlay)

                # 1-D CAM → save curve for later plotting
                elif cam.ndim == 1:                   # (T,)
                    np.save(layer_dir / "temporal_cam.npy", cam)
                else:
                    print(f"[WARN] Unknown CAM shape {cam.shape} – skipped save")

            # also compute model predictions
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(*inp) if isinstance(inp, list) else model(inp)
                pred = out.argmax(-1)
                preds[name] = " ".join(str(int(p)) for p in pred.squeeze())

        # write ground-truth and predictions to file
        with open(outputs_dir / "preds.txt", "w") as f:
            gt = " ".join(str(int(x)) for x in batch["gloss"])
            f.write(f"GT: {gt}\n")
            for model_name, text in preds.items():
                f.write(f"{model_name}: {text}\n")

    print("[DONE] outputs written to", out_root)


if __name__ == "__main__":
    main()