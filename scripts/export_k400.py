#!/usr/bin/env python3
"""Save the official torchvision R(2+1)D-18 K400 state dict locally."""

import argparse
import hashlib
from pathlib import Path

import torch
from torchvision.models.video import R2Plus1D_18_Weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    state = R2Plus1D_18_Weights.KINETICS400_V1.get_state_dict(progress=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Saved {args.output} (SHA256: {digest})")


if __name__ == "__main__":
    main()
