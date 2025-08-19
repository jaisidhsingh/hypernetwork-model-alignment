<div align="center">

# (Almost) Free Modality Stitching of Foundation Models

Jaisidh Singh $^{1,2,4}$, Diganta Misra $^{3,4}$, Boris Knyazev $^{5}$, Antonio Orvieto $^{3,4,6}$

$^1$ University of Tübingen, $^2$ Zuse School ELIZA, $^3$ ELLIS Institute Tübingen, $^4$ MPI-IS Tübingen, $^5$ SAIT AI Lab Montréal, $^6$ Tübingen AI Center 

<a href="https://arxiv.org/pdf/2507.10015">![Paper](https://img.shields.io/badge/paper-arxiv.2507.10015-red)</a>
![License](https://img.shields.io/badge/license-MIT-blue.svg)

### Abstract

Foundation multi-modal models are often designed by stitching of multiple existing pretrained uni-modal models: for example, an image classifier with an text model. This stitching process is performed by training a connector module that aims to align the representation spaces of these uni-modal models towards a multi-modal objective. However, given the complexity of training such connectors on large scale web-based datasets coupled with the ever-increasing number of available pretrained uni-modal models, the task of uni-modal models selection and subsequent connector module training becomes computationally demanding. To address this under-studied critical problem, we propose Hypernetwork Model Alignment (Hyma), a novel all-in-one solution for optimal uni-modal model selection and connector training by leveraging hypernetworks. Specifically, our framework utilizes the parameter prediction capability of a hypernetwork to obtain jointly trained connector modules for $N\times M$ combinations of uni-modal models. In our experiments, Hyma reduces the cost of searching for the best performing uni-modal model pair by $10\times$, while matching the ranking and trained connector performance obtained via grid search across a suite of diverse multi-modal benchmarks.

<img src="./assets/overview_new3-3-1.png">
</div>

<br>
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