# main.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

from dataset import PhoenixVideoTextDataset
from cam_runner import CAMRunner, save_or_overlay   # ← new helper
from models.slowfast_sign import SlowFastSign
from models.twostream_slr import TwoStreamSLR


# ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", help="single sequence id or comma-separated list")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--models", nargs="*", default=["slowfast"],
                   choices=["slowfast", "twostream"])
    p.add_argument("--slowfast-ckpt")
    p.add_argument("--twostream-ckpt")
    p.add_argument("--target-from-gt", action="store_true",
                   help="use middle-frame ground-truth gloss for CAM target")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ────────────────────────────────────────────────────────────
def load_model(name: str, ckpt_path: str | None) -> torch.nn.Module:
    model = SlowFastSign() if name == "slowfast" else TwoStreamSLR()
    if ckpt_path and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model


# ────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device)

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "data" / "phoenix-2014-multisigner"
    out_root  = repo_root / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    seq_list: List[str] | None = (
        [s.strip() for s in args.seq.split(",")] if args.seq else None
    )
    dataset = PhoenixVideoTextDataset(data_root, args.split, seq_ids=seq_list)

    for i in range(len(dataset)):
        batch   = dataset[i]
        seq_id  = dataset.seq_ids[i]
        seq_out = out_root / seq_id
        seq_out.mkdir(parents=True, exist_ok=True)

        preds: Dict[str, str] = {}
        for name in args.models:
            ckpt  = args.slowfast_ckpt if name == "slowfast" else args.twostream_ckpt
            model = load_model(name, ckpt).to(device)

            # -------- build 5-D inputs (B,C,T,H,W) --------
            rgb5d = batch["rgb"].permute(1, 0, 2, 3).unsqueeze(0).to(device)
            if name == "slowfast":
                inp = rgb5d
            else:
                if batch["pose"] is not None:
                    pose5d = batch["pose"].permute(1, 0, 2, 3).unsqueeze(0).to(device)
                else:   # fallback: zero pose
                    B, C, T, H, W = rgb5d.shape
                    pose5d = torch.zeros((B, 2, T, H, W), dtype=rgb5d.dtype, device=device)
                inp = [rgb5d, pose5d]

            # -------- run CAM extraction ------------------
            runner = CAMRunner(model, model.target_layers)
            target = None
            if args.target_from_gt and len(batch["gloss"]) > 0:
                target = int(batch["gloss"][len(batch["gloss"]) // 2])
            cams = runner.run(inp, device, target)

            # -------- save CAMs ---------------------------
            for idx, cam in enumerate(cams):
                layer_dir = seq_out / name / f"layer{idx + 1}"
                layer_dir.mkdir(parents=True, exist_ok=True)

                # spatial CAM (T,H,W)  or  (H,W)→(1,H,W)
                if cam.ndim == 2:
                    cam = cam[None, ...]

                for t in range(min(cam.shape[0], len(batch["rgb"]))):
                    frame = batch["rgb"][t].permute(1, 2, 0).cpu().numpy()
                    save_or_overlay(frame, cam[t], layer_dir, t)

            # -------- predictions -------------------------
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(*inp) if isinstance(inp, list) else model(inp)
            preds[name] = " ".join(str(int(x)) for x in out.argmax(-1).flatten())

        # save GT + predictions
        with open(seq_out / "preds.txt", "w") as f:
            f.write("GT:  " + " ".join(str(int(x)) for x in batch["gloss"]) + "\n")
            for m, p in preds.items():
                f.write(f"{m}: {p}\n")

    print("[DONE] outputs written to", out_root)


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
