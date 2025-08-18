import os
import sys
import torch
import numpy as np
from huggingface_hub import upload_file

img_data_dir = "/fast/jsingh/hyperalign/results/image_embeddings"
txt_data_dir = "/fast/jsingh/hyperalign/results/text_embeddings"
ie_name_maps = {
    384: {
        "vit_small_patch16_224.augreg_in21k_ft_in1k": "vit_s16",
        "deit_small_patch16_224.fb_in1k": "deit_s16",
        "deit3_small_patch16_224.fb_in1k": "deit3_s16"
    },
    768: {
        "deit3_base_patch16_224.fb_in22k_ft_in1k": "deit3-ft_b16",
        "deit_base_patch16_224.fb_in1k": "deit_b16",
        "vit_base_patch16_224.augreg_in21k_ft_in1k": "vit_b16"
    },
    1024: {
        "deit3_large_patch16_384.fb_in22k_ft_in1k": "deit3-ft_l16",
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k": "eva2-ft_l16",
        "vit_large_patch16_224.augreg_in21k_ft_in1k": "vit_l16"
    }
}
te_name_maps = {
    384: {
        "all-MiniLM-L12-v2": "minilm"
    },
    768: {
        "all-mpnet-base-v2": "mpnet"
    },
    1024: {
        "all-roberta-large-v1": "roberta"
    }
}

def upload(dim, mode):
    if mode == "img":
        ref = ie_name_maps
        data_dir = img_data_dir
    elif mode == "txt":
        ref = te_name_maps
        data_dir = txt_data_dir

    for k, v in ref[dim].items():
        pt_file_path = f"{data_dir}/dim_{dim}/{k}/cc3m558k.pt"
        if not os.path.exists(pt_file_path):
            np_file_path = pt_file_path.replace(".pt", ".npy")
            np_file = np.load(np_file_path, allow_pickle=True)
            torch.save(torch.from_numpy(np_file), pt_file_path)
        
        if mode == "img":
            repo_id = f"jaisidhsingh/cc3m558k-img-embed-dim-{dim}"
        else:
            repo_id = "jaisidhsingh/cc3m558k-txt-embed-all-dims"

        upload_file(
            path_or_fileobj=pt_file_path,
            path_in_repo=f"{v}.pt",
            repo_id=repo_id,
            repo_type="dataset"
        )


if __name__ == "__main__":
    dim = int(sys.argv[1])
    mode = str(sys.argv[2])
    upload(dim, mode)
