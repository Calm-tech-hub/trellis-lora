import argparse
import os
import warnings

import torch
from PIL import Image

import utils3d

from trellis.pipelines import TrellisImageTo3DPipeline

os.environ['SPCONV_ALGO'] = 'native'

warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")


def save_sparse_structure_as_ply(coords: torch.Tensor, path: str, resolution: int = 64) -> None:
    coords = coords[coords[:, 0] == 0][:, 1:].float().cpu()
    points = (coords + 0.5) / resolution - 0.5
    utils3d.io.write_ply(path, points.numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Save TRELLIS stage-1 sparse structure as PLY')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output PLY path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--steps', type=int, default=12, help='Stage-1 sampling steps')
    parser.add_argument('--cfg-strength', type=float, default=7.5, help='Stage-1 CFG strength')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
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
                "steps": args.steps,
                "cfg_strength": args.cfg_strength,
            },
        )

    save_sparse_structure_as_ply(coords, args.output)


if __name__ == '__main__':
    main()
