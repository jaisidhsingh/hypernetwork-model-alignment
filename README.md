<div align="center">

# (Almost) Free Modality Stitching of Foundation Models
<a href="https://arxiv.org/pdf/2507.10015">![Paper](https://img.shields.io/badge/paper-arxiv.2507.10015-red)</a>
![License](https://img.shields.io/badge/license-MIT-blue.svg)

<img src="./assets/overview_new3-3-1.png">
</div>

This repositories releases the data used to train a generative model of multi-modal connectors as well as the checkpoints of the multi-modal connectors presented in the paper "(Almost) Free Modality Stitching of Foundation Models".

## Checkpoints

The checkpoints of $\text{MLP}_1$ connectors for VLMs obtained via our method HYMA and Grid Search are uploaded to <a href="https://huggingface.co/collections/jaisidhsingh/hyma-vlm-connector-checkpoints-68a34befaad027913f605c81">here</a>. Note that we upload the connectors for the best performing encoder pairs, of feature dimension 1024 and 768 (for both image and text). Each `.pt` filename provides the description of the image encoder and text encoder aligned via the connector.

## Pre-emebedded CC3M-558K dataset

We embed the samples of CC3M-558K (in-order) across 9 image encoders and 3 text encoders, normalize all embeddings, and upload them <a href="https://huggingface.co/collections/jaisidhsingh/hyma-llava-alignment-cc3m-558k-pre-embedded-68a34597f1e8d93e2a40c8b4">here</a>. Each `.pt` filename provides the description of the model used to embed the data.

## Citation

If you found our work useful, please cite our paper as
```tex
@article{singh2025almost,
  title={(Almost) Free Modality Stitching of Foundation Models},
  author={Singh, Jaisidh and Misra, Diganta and Knyazev, Boris and Orvieto, Antonio},
  journal={arXiv preprint arXiv:2507.10015},
  year={2025}
}
```

## Todos

- [x] Upload data
- [x] Upload checkpoints
- [ ] Add code to load in the VLM into a `CustomVLM` object
- [ ] Add code to load in the data into a `Dataset` object
- [x] Add more instructions to README 
- [ ] Add model cards and data cards on huggingface.
- [ ] Add citation