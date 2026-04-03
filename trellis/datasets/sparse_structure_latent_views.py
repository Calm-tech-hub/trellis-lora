import json
import os
from typing import *

import numpy as np
import torch
from PIL import Image

from .components import StandardDatasetBase
from .sparse_structure_latent import SparseStructureLatentVisMixin


class SparseStructureLatentViews(SparseStructureLatentVisMixin, StandardDatasetBase):
    def __init__(
        self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        num_views: int = 12,
        image_size: int = 518,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.num_views = num_views
        self.image_size = image_size
        self.normalization = normalization
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['cond_rendered'] == True]
        stats['Cond rendered'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_hsize = hsize * 1.2
        aug_bbox = [
            int(center[0] - aug_hsize),
            int(center[1] - aug_hsize),
            int(center[0] + aug_hsize),
            int(center[1] + aug_hsize),
        ]
        image = image.crop(aug_bbox)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        return image * alpha.unsqueeze(0)

    def get_instance(self, root, instance):
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = min(self.num_views, len(metadata['frames']))
        view = np.random.randint(n_views)
        frame = metadata['frames'][view]

        image_path = os.path.join(image_root, frame['file_path'])
        latent_path = os.path.join(root, 'ss_latents', self.latent_model, instance, f'{view:03d}.npz')

        image = self._load_image(image_path)
        latent = np.load(latent_path)
        z = torch.tensor(latent['mean']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        return {
            'x_0': z,
            'cond': image,
        }


class ImageConditionedSparseStructureLatentViews(SparseStructureLatentViews):
    pass
