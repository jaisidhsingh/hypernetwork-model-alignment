import os
import torch
import torch.nn as nn
from upload_data import ie_name_maps, te_name_maps


ie_index_map = {
    384: [0, 1, 2],
    768: [4, 3, 0],
    1024: [2, 3, 0]
}
ie_config = {
    384: [
		"vit_small_patch16_224.augreg_in21k_ft_in1k",
		"deit_small_patch16_224.fb_in1k",
		"deit3_small_patch16_224.fb_in1k",
		"flexivit_small.300ep_in1k", # unused
		"eva02_small_patch14_224.mim_in22k", # unused
	],
    768: [
		"vit_base_patch16_224.augreg_in21k_ft_in1k",
		"vit_base_patch32_224.augreg_in21k_ft_in1k", # unused
		"vit_base_patch32_clip_224.laion2b_ft_in12k_in1k", # unused
		"deit_base_patch16_224.fb_in1k",
		"deit3_base_patch16_224.fb_in22k_ft_in1k",
	],
	1024: [
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
		"vit_large_patch14_clip_336.laion2b_ft_in12k_in1k", # unused
		"deit3_large_patch16_384.fb_in22k_ft_in1k",
		"eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
		"beitv2_large_patch16_224.in1k_ft_in22k_in1k" # unused
	]
}
te_config = {
	384: ["all-MiniLM-L12-v2"],
	768: ["all-mpnet-base-v2"],
	1024: ["all-roberta-large-v1"]
}


def load_ape_ckpt(ied, iei, ted, c):
    name = f"c{c}_ie-{ied}-{iei}_te-{ted}-0"
    path = f"/fast/jsingh/hyperalign/checkpoints/vlms/ape/dim_384/{name}/seed_0/ckpt_350.pt"
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    return ckpt["model"]

def load_hnet_ckpt(ied, iei, ted, c):
    idx = ie_index_map[ied].index(iei)
    name = f"c{c}_ie-{ied}_te{ted}"
    path = f"/fast/jsingh/hyperalign/checkpoints/vlms/hnet/{name}/seed_1/ckpt_10.pt"
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    return ckpt["params"][idx]


def load_model(args, step, mode="ape"):
    (ie_dim, ie_index, te_dim, te_index, C, seed) = args
    if C == 0:
        model = nn.Linear(te_dim, ie_dim).cpu()

    if C == 1:
        model = nn.Sequential(
            nn.Linear(te_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, ie_dim),
        ).cpu()

    if C == 2:
        model = nn.Sequential(
            nn.Linear(te_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, ie_dim),
        ).cpu()

    if mode == "ape":
        folder = f"/fast/jsingh/hyperalign/checkpoints/vlms/ape/dim_384/c{C}_ie-{ie_dim}-{ie_index}_te-{te_dim}-{te_index}/seed_{seed}"
        ckpt = torch.load(os.path.join(folder, f"ckpt_{step}.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    
    else:
        folder = f"/fast/jsingh/hyperalign/checkpoints/vlms/hnet/c{C}_ie-{ie_dim}_te{te_dim}/seed_{seed}"
        full_ckpt = torch.load(os.path.join(folder, f"ckpt_{step}.pt"), map_location="cpu", weights_only=False)

        idx_map = {
            384: {i:i for i in range(3)},
            768: {j : i for i, j in zip(range(3), [4, 3, 0])},
            1024: {j : i for i, j in zip(range(3), [2, 3, 0])}
        }
        index = idx_map[ie_dim][ie_index]
        ckpt = full_ckpt["params"][index]

        if C == 0:
            jj = 0
            ckpt[f"weight"] = ckpt[f"layers.{jj}.weight"]
            ckpt[f"bias"] = ckpt[f"layers.{jj}.bias"]

            ckpt.pop(f"layers.{jj}.weight") 
            ckpt.pop(f"layers.{jj}.bias")
            
        if C == 1:
            for jj in [0, 2]:
                ckpt[f"{jj}.weight"] = ckpt[f"layers.{jj}.weight"]
                ckpt[f"{jj}.bias"] = ckpt[f"layers.{jj}.bias"]
                ckpt.pop(f"layers.{jj}.weight") 
                ckpt.pop(f"layers.{jj}.bias")
        
        if C == 2 and "layers.4.weight" not in ckpt:
            for jj in [0, 2]:
                ckpt[f"{jj}.weight"] = ckpt[f"layers.{jj}.weight"]
                ckpt[f"{jj}.bias"] = ckpt[f"layers.{jj}.bias"]

                ckpt.pop(f"layers.{jj}.weight") 
                ckpt.pop(f"layers.{jj}.bias")
            
            jj = 3
            
            ckpt[f"4.weight"] = ckpt[f"layers.{jj}.weight"]
            ckpt[f"4.bias"] = ckpt[f"layers.{jj}.bias"]
            ckpt.pop(f"layers.{jj}.weight") 
            ckpt.pop(f"layers.{jj}.bias")
        
        model.load_state_dict(ckpt)

    model.eval() 
    return model.state_dict()

def save_vlm(ied, iei, ted, c, mode):
    seed = 0 if mode == "ape" else 1
    step = 350 if mode == "ape" else 10

    args = (ied, iei, ted, 0, c, seed)
    ckpt = load_model(args, step, mode)

    ie_name = ie_name_maps[ied][ie_config[ied][iei]]
    te_name = te_name_maps[ted][te_config[ted][0]]

    save_name = f"{ie_name}_mlp{c}_{te_name}.pt"

    save_path = f"/fast/jsingh/hyperalign/release_checkpoints/{mode}/{save_name}"
    torch.save(ckpt, save_path)
    print(f"Saved VLM to {save_path}")


if __name__ == "__main__":
    mode = "ape"
    dims = [768, 1024]
    for ied in dims:
        for iei in ie_index_map[ied]:
            for ted in dims:
                for c in [1]:
                    save_vlm(ied, iei, ted, c, mode)
