"""
Train U-Net on the Kaggle LGG MRI Segmentation dataset
"""
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from unet import UNet, preprocess

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_3m"
EPOCHS, BATCH, LR = 25, 16, 1e-3
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class MRI(Dataset):
    def __init__(self, masks, augment):
        self.masks, self.augment = masks, augment

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i):
        m = self.masks[i]
        x = preprocess(Image.open(m.replace("_mask", "")))
        y = np.asarray(Image.open(m).convert("L").resize((256, 256), Image.NEAREST))
        y = torch.from_numpy((y > 0).astype(np.float32))[None]
        if self.augment and random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def dice(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    return (2 * (p * y).sum((1, 2, 3)) + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)


def main():
    masks = sorted(glob.glob(f"{DATA}/*/*_mask.tif"))
    assert masks, f"No masks found under {DATA}"
    patients = sorted({os.path.dirname(m) for m in masks})
    random.seed(0)
    random.shuffle(patients)
    val_p = set(patients[: len(patients) // 5])
    tr = [m for m in masks if os.path.dirname(m) not in val_p]
    va = [m for m in masks if os.path.dirname(m) in val_p]
    print(f"{len(tr)} train slices, {len(va)} val slices, device={dev}")

    tl = DataLoader(MRI(tr, True), BATCH, shuffle=True, num_workers=2)
    vl = DataLoader(MRI(va, False), BATCH, num_workers=2)
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    best = 0.0

    for ep in range(EPOCHS):
        net.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            out = net(x)
            loss = F.binary_cross_entropy_with_logits(out, y) + (1 - dice(out, y)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        d, iou, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                pred = (torch.sigmoid(net(x)) > 0.5).float()
                inter = (pred * y).sum((1, 2, 3))
                union = pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) - inter
                d += ((2 * inter + 1) / (pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).sum().item()
                iou += ((inter + 1) / (union + 1)).sum().item()
                n += len(x)
        d, iou = d / n, iou / n
        print(f"epoch {ep + 1:02d}  loss {loss.item():.3f}  val dice {d:.3f}  val iou {iou:.3f}")
        if d > best:
            best = d
            torch.save(net.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
