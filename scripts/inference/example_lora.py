import argparse
import os
import warnings

import imageio
import torch
from PIL import Image

from trellis.modules.lora import apply_lora_to_linear_layers
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils


os.environ['SPCONV_ALGO'] = 'native'

warnings.filterwarnings('ignore', category=FutureWarning, module='spconv')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run TRELLIS image-to-3D inference with optional LoRA checkpoints')
    parser.add_argument('--image', type=str, default=None, help='Input image path')
    parser.add_argument('--images', type=str, nargs='+', default=None, help='Input image paths for multi-image inference')
    parser.add_argument('--stage1-ckpt', type=str, default=None, help='Optional LoRA-adapted stage1 checkpoint path')
    parser.add_argument('--stage2-ckpt', type=str, default=None, help='Optional LoRA-adapted stage2 checkpoint path')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--ss-steps', type=int, default=12, help='Stage-1 sampling steps')
    parser.add_argument('--ss-cfg-strength', type=float, default=7.5, help='Stage-1 CFG strength')
    parser.add_argument('--slat-steps', type=int, default=12, help='Stage-2 sampling steps')
    parser.add_argument('--slat-cfg-strength', type=float, default=3.0, help='Stage-2 CFG strength')
    parser.add_argument('--multiimage-mode', type=str, default='multidiffusion', choices=['stochastic', 'multidiffusion'], help='Multi-image sampling mode')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank used during training')
    parser.add_argument('--alpha', type=float, default=8.0, help='LoRA alpha used during training')
    parser.add_argument('--dropout', type=float, default=0.0, help='LoRA dropout used during training')
    args = parser.parse_args()
    if args.image is None and not args.images:
        parser.error('Either --image or --images must be provided')
    if args.image is not None and args.images:
        parser.error('Use either --image or --images, not both')
    return args


def apply_lora(model, rank: int, alpha: float, dropout: float) -> None:
    apply_lora_to_linear_layers(
        model,
        target_keywords=['self_attn.to_qkv', 'self_attn.to_out', 'cross_attn.to_q', 'cross_attn.to_kv', 'cross_attn.to_out'],
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')

    stage1 = pipeline.models['sparse_structure_flow_model']
    if args.stage1_ckpt is not None:
        apply_lora(stage1, args.rank, args.alpha, args.dropout)
        stage1_state_dict = torch.load(args.stage1_ckpt, map_location='cpu', weights_only=True)
        stage1.load_state_dict(stage1_state_dict, strict=True)

    stage2 = pipeline.models['slat_flow_model']
    if args.stage2_ckpt is not None:
        apply_lora(stage2, args.rank, args.alpha, args.dropout)
        stage2_state_dict = torch.load(args.stage2_ckpt, map_location='cpu', weights_only=True)
        stage2.load_state_dict(stage2_state_dict, strict=True)

    pipeline.cuda()

    sparse_structure_sampler_params = {
        'steps': args.ss_steps,
        'cfg_strength': args.ss_cfg_strength,
    }
    slat_sampler_params = {
        'steps': args.slat_steps,
        'cfg_strength': args.slat_cfg_strength,
    }

    if args.images:
        images = [Image.open(image_path) for image_path in args.images]
        outputs = pipeline.run_multi_image(
            images,
            seed=args.seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
            mode=args.multiimage_mode,
        )
    else:
        image = Image.open(args.image)
        outputs = pipeline.run(
            image,
            seed=args.seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            slat_sampler_params=slat_sampler_params,
        )

    video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
    imageio.mimsave(os.path.join(args.output_dir, 'sample_gs.mp4'), video_gs, fps=30)

    video_rf = render_utils.render_video(outputs['radiance_field'][0], bg_color=(1, 1, 1))['color']
    imageio.mimsave(os.path.join(args.output_dir, 'sample_rf.mp4'), video_rf, fps=30)

    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(args.output_dir, 'sample_mesh.mp4'), video_mesh, fps=30)

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(os.path.join(args.output_dir, 'sample.glb'))
    outputs['gaussian'][0].save_ply(os.path.join(args.output_dir, 'sample_gs.ply'))


if __name__ == '__main__':
    main()
