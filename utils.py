import timm
import torch
from typing import *
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from upload_data import ie_name_maps, te_name_maps
from sentence_transformers import SentenceTransformer


def reverse_name_maps(name_map):
    result = {}
    for d, c in name_map.items():
        result[d] = {}
        for k, v in c.items():
            result[d][v] = k
    return result


def find_name_in_map(name, name_map):
    for d, c in name_map.items():
        if name in c.keys():
            return c[name], d


class ImageEncoder(nn.Module):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device

        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.config = timm.data.resolve_model_data_config(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = timm.data.create_transform(**self.config, is_training=False)

    def encode_image(self, image):
        image_features = self.model(image)
        return F.normalize(image_features, dim=-1)

    def forward(self, image):
        return self.encode_image(image)


class TextEncoder():
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = device

        self.model = SentenceTransformer(model_name).to(device)
        self.model.eval()

    def to(self, x):
        self.model.to(x)

    def __call__(self, x):
        return self.encode_text(x)

    def encode_text(self, sentences):
        text_features = self.model.encode(sentences)
        text_features = torch.from_numpy(text_features).to(self.device)
        return F.normalize(text_features, dim=-1)


def init_modality_connector(input_dim, hidden_dims, output_dim, activation="relu"):
    if activation == "relu":
        act_fn = nn.ReLU()
    elif activation == "gelu":
        act_fn = nn.GELU()
    
    num_layers = len(hidden_dims) + 1
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    layers = [nn.Linear(input_dim, layer_dims[1])]
    
    for i in range(0, num_layers):
        if i % 2 != 0:
            layers.append(act_fn)
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
    
    return nn.Sequential(*layers)


class CustomVLM():
    def  __init__(self, image_encoder_name, text_encoder_name, connector_args=None, device="cuda"):
        self.image_encoder = ImageEncoder(image_encoder_name)
        self.image_encoder.model = self.image_encoder.model.to(device)

        self.text_encoder = TextEncoder(text_encoder_name)
        self.text_encoder.model = self.text_encoder.model.to(device)

        self.connector = None
        if connector_args is not None:
            self.connector = init_modality_connector(*connector_args).to(device)
            self.connector.eval()
        
        self.device = device
    
    def load_connector_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.connector.load_state_dict(checkpoint)

    def encode_image(self, x):
        return self.image_encoder.encode_image(x)

    def encode_text(self, x):
        x = self.text_encoder.encode_text(x)
        if self.connector is not None:
            x = self.connector(x)
        return x


class EmbeddingDataset(Dataset):
    def __init__(self, filepath):
        self.embeddings = torch.load(filepath, weights_only=False, map_location="cpu")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]   


def get_vlm_from_checkpoint(checkpoint_path):
    filename = checkpoint_path.split("/")[-1]

    IMAGE_ENCODER_CONFIG = reverse_name_maps(ie_name_maps)
    TEXT_ENCODER_CONFIG = reverse_name_maps(te_name_maps)

    ie_name, te_name = filename.split("_mlp1_")[0], [-1]
    (ie_name_full, ie_dim), (te_name_full, te_dim) = find_name_in_map(ie_name, IMAGE_ENCODER_CONFIG), find_name_in_map(te_name, TEXT_ENCODER_CONFIG)

    vlm_kwargs = {
        "image_encoder_name": ie_name_full,
        "text_encoder_name": te_name_full,
        "connector_args": (te_dim, [1024], ie_dim)
    }
    model = CustomVLM(**vlm_kwargs)
    model.load_connector_checkpoint(checkpoint_path)
    return model