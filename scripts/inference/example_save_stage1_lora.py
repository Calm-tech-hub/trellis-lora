import argparse
import os
import warnings

import torch
from PIL import Image

import utils3d

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.lora import apply_lora_to_linear_layers


os.environ['SPCONV_ALGO'] = 'native'

warnings.filterwarnings('ignore', category=FutureWarning, module='spconv')


def save_sparse_structure_as_ply(coords: torch.Tensor, path: str, resolution: int = 64) -> None:
    coords = coords[coords[:, 0] == 0][:, 1:].float().cpu()
    points = (coords + 0.5) / resolution - 0.5
    utils3d.io.write_ply(path, points.numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Save TRELLIS LoRA-adapted stage-1 sparse structure as PLY')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output PLY path')
    parser.add_argument('--ckpt', type=str, required=True, help='LoRA-adapted stage1 checkpoint path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--steps', type=int, default=12, help='Stage-1 sampling steps')
    parser.add_argument('--cfg-strength', type=float, default=7.5, help='Stage-1 CFG strength')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank used during training')
    parser.add_argument('--alpha', type=float, default=8.0, help='LoRA alpha used during training')
    parser.add_argument('--dropout', type=float, default=0.0, help='LoRA dropout used during training')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
    denoiser = pipeline.models['sparse_structure_flow_model']
    apply_lora_to_linear_layers(
        denoiser,
        target_keywords=['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q', 'cross_attn.to_kv', 'cross_attn.to_out'],
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    denoiser.load_state_dict(state_dict, strict=True)
    pipeline.cuda()

    with torch.no_grad():
        image = Image.open(args.image)
        image = pipeline.preprocess_image(image)
        cond = pipeline.get_cond([image])

        torch.manual_seed(args.seed)
        coords = pipeline.sample_sparse_structure(
            cond,
            num_samples=1,
            sampler_params={
                'steps': args.steps,
                'cfg_strength': args.cfg_strength,
            },
        )

    save_sparse_structure_as_ply(coords, args.output)


if __name__ == '__main__':
    main()
