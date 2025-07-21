from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from dataset import PhoenixVideoTextDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='train')
    p.add_argument('--seq')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent.parent
    root = base/ "data" / "phoenix-2014-multisigner"
    seqs = [args.seq] if args.seq else []
    if not seqs:
        csv = root / f"annotations/manual/{args.split}.corpus.csv"
        if csv.exists():
            with open(csv) as f:
                seqs = [line.split("|")[0] for line in f.read().splitlines()][1:]
    for seq in tqdm(seqs):
        ds = PhoenixVideoTextDataset(root, args.split, [seq])
        _ = ds[0]

if __name__ == "__main__":
    main()