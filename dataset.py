from __future__ import annotations
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class PhoenixVideoTextDataset(Dataset):
    ''' PHOENIX-2014 video + text loader with optional pose heat-maps.'''

    def __init__(self,
                 root: Path,
                 split: str,
                 seq_ids: Optional[List[str]] = None,
                 frame_skip: int = 1,
                 crop_size: int = 224,
                 pose_cache_dir: Optional[Path] = None,
                 ):
        self.root = Path(root)
        self.split = split
        self.frame_skip = frame_skip
        self.crop_size = crop_size
        self.pose_cache_dir = (
            Path(pose_cache_dir) if pose_cache_dir is not None else self.root / "keypoints" / "heatmaps"
        )
        self.seq_ids, self.seq_tokens = self._load_seq_table(seq_ids)
        self.id2gloss = self._load_id_map()

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(self.crop_size),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


    def _load_seq_table(self, seq_ids: Optional[List[str]]):
        '''    
        Args:
          seq_ids: Optional list of sequence IDs to filter on.
        Operation:
          Reads the split-specific CSV annotation file, parses each line into
          a sequence ID and its tokenized glosses. If seq_ids is provided,
          filters the loaded IDs to those present in seq_ids.
        Returns:
          A tuple containing:
            - List[str]: the ordered sequence IDs to use.
            - Dict[str, List[str]]: a mapping from each sequence ID to its list of gloss tokens.
        '''
        
        csv_path = self.root / "annotations" / "manual" / f"{self.split}.corpus.csv"
        all_ids: List[str] = []
        tokens: Dict[str, List[str]] = {}
        with open(csv_path) as f:
            reader = csv.reader(f, delimiter='|')
            next(reader)  # skip header
            for row in reader:
                seq = row[0]
                glosses = row[1].split()
                all_ids.append(seq)
                tokens[seq] = glosses
        if seq_ids is not None:
            seq_ids = [s for s in seq_ids if s in tokens]
            return seq_ids, {k: tokens[k] for k in seq_ids}
        return all_ids, tokens


    def _load_id_map(self):
        '''
        Args: None
        Operation:
        Loads or constructs the mapping from numeric gloss IDs back to string labels
        by unpickling "gloss2ids.pkl" and inverting the dictionary.
        Returns:
        Dict[int, str]: a mapping from gloss ID to gloss string. Empty if file not found.
        '''

        pkl = self.root / "annotations" / "manual" / "gloss2ids.pkl"
        if pkl.exists():
            mapping = _load_pickle(pkl)
            return {v: k for k, v in mapping.items()}
        return {}

   
    def __len__(self) -> int:
        '''
        Args:
        index: integer index into the dataset.
        Operation:
        Returns the total number of sequences loaded.
        Returns:
        int: the length of the dataset (number of sequences).
        '''

        return len(self.seq_ids)


    def __getitem__(self, index: int):
        '''
        Args:
        index: integer index of the sequence to load.
        Operation:
        Reads frames from disk (with skipping), applies transforms to RGB,
        loads or computes pose heatmaps, tokenizes glosses and converts to IDs.
        Returns:
        Dict[str, Any]: a dictionary with keys:
            - "rgb": Tensor of shape (T, 3, H, W)
            - "pose": Optional Tensor of shape (T, 2, H, W) or None
            - "gloss": LongTensor of gloss IDs
            - "length": int total number of raw frames considered
        '''

        seq = self.seq_ids[index]
        frames_dir = (
            self.root / "features" / "fullFrame-256x256px" / self.split / seq / "1"
        )
        frames_paths = sorted(frames_dir.glob("*.png"))[::self.frame_skip]
        rgb_imgs = [
            cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            for p in frames_paths
        ]
        rgb = torch.stack([self.rgb_transform(img) for img in rgb_imgs])

        pose = self._load_pose(seq, len(frames_paths))

        tokens = self.seq_tokens.get(seq, [])
        gloss = (
            torch.tensor([self._gloss_id(t) for t in tokens], dtype=torch.long)
            if tokens else torch.LongTensor([])
        )
        return {
            "rgb": rgb,
            "pose": pose,
            "gloss": gloss,
            "length": len(frames_paths) * self.frame_skip,
        }


    def _gloss_id(self, token: str) -> int:
        '''
        Args:
        token: a single gloss string.
        Operation:
        Looks up the corresponding numeric gloss ID in the inverted map.
        Returns:
        int: the gloss ID, or 0 if not found.
        '''

        for idx, gloss in self.id2gloss.items():
            if gloss == token:
                return idx
        return 0


    def _load_pose(self, seq: str, num_frames: int) -> Optional[torch.Tensor]:
        '''
        Args:
        seq: sequence ID string
        num_frames: number of frames expected after skipping
        Operation:
        Attempts to load pose heatmaps from cache; if missing, loads raw keypoints,
        renders them into heatmaps, caches the result.
        Returns:
        Optional[Tensor]: FloatTensor of shape (num_frames, 2, H, W) or None if no keypoints file.
        '''

        keypkl = self.root / "keypoints" / "keypoints.pkl"
        if not keypkl.exists():
            return None
        cache = self.pose_cache_dir / f"{seq}.npy"
        if cache.exists():
            data = np.load(cache)
        else:
            # kp = _load_pickle(keypkl)[seq][::self.frame_skip]

            # key inside .pkl uses the 210×260 path
            seq210 = seq.replace("fullFrame-256x256px", "fullFrame-210x260px")
            kp = _load_pickle(keypkl)[seq210]["keypoints"][::self.frame_skip]
            data = self._render_heatmaps(np.asarray(kp, dtype=np.float32))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, data)
        return torch.from_numpy(data[:num_frames]).float()


    def _render_heatmaps(self, keypoints: np.ndarray) -> np.ndarray:
        '''
        Args:
        keypoints: ndarray with shape (T, K, 3) where each entry is (x, y, confidence)
        Operation:
        For each frame and keypoint, draws a small circle in one of two channels
        based on the keypoint index parity, then resizes each heatmap to crop_size.
        Returns:
        ndarray: shape (T, 2, crop_size, crop_size) of float32 heatmaps.
        '''

        t = keypoints.shape[0]
        hm = np.zeros((t, 2, 112, 112), dtype=np.float32)
        for i in range(t):
            for j, (x, y, c) in enumerate(keypoints[i]):
                if c <= 0:
                    continue
                x_i = int(round(x / 256 * 112))
                y_i = int(round(y / 256 * 112))
                if not (0 <= x_i < 112 and 0 <= y_i < 112):
                    continue
                channel = 0 if j % 2 == 0 else 1
                cv2.circle(hm[i, channel], (x_i, y_i), 4, 1.0, -1)
        resized = [
            cv2.resize(h, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            for h in hm.reshape(-1, 112, 112)
        ]
        return np.stack(resized).reshape(t, 2, self.crop_size, self.crop_size)