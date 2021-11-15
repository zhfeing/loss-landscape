import argparse
import os
from collections import OrderedDict

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutual-ckpt", type=str)
    parser.add_argument("--out-dir", type=str)
    args = parser.parse_args()

    ckpt_fp = os.path.expanduser(args.mutual_ckpt)
    ckpt = torch.load(ckpt_fp, map_location="cpu")
    ckpt = ckpt["model"]

    ckpt_cnn = OrderedDict()
    ckpt_vit = OrderedDict()
    for k, v in ckpt.items():
        kk = k.split(".")
        model_name = kk[1]
        new_k = ".".join(kk[2:])
        if model_name == "cnn":
            ckpt_cnn[new_k] = v
        elif model_name == "vit":
            ckpt_vit[new_k] = v
        else:
            raise Exception("Unknown model name: {}".format(model_name))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(ckpt_cnn, os.path.join(args.out_dir, "cnn.pth"))
    torch.save(ckpt_vit, os.path.join(args.out_dir, "vit.pth"))
