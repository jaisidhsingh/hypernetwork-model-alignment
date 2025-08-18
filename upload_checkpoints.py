import os
import sys
import torch
import numpy as np
from huggingface_hub import upload_file

repo_id = lambda x: f"jaisidhsingh/hyma4vlms-{x}-mlp1"

def hnet():
    folder = "/fast/jsingh/hyperalign/release_checkpoints/hnet"
    files = os.listdir(folder)
    paths = [os.path.join(folder, f) for f in files]

    for f, p in zip(files, paths):
        upload_file(
            path_or_fileobj=p,
            path_in_repo=f,
            repo_id=repo_id("ours"),
            repo_type="model"
        )


def ape():
    folder = "/fast/jsingh/hyperalign/release_checkpoints/ape"
    files = os.listdir(folder)
    paths = [os.path.join(folder, f) for f in files]

    for f, p in zip(files, paths):
        upload_file(
            path_or_fileobj=p,
            path_in_repo=f,
            repo_id=repo_id("gs"),
            repo_type="model"
        )


if __name__ == "__main__":
    mode = str(sys.argv[1])
    if mode == "hnet":
        hnet()
    else:
        ape()
