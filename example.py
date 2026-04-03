import argparse
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="spconv")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the base TRELLIS pipeline')
    parser.add_argument('--image', type=str, default='assets/example_image/typical_vehicle_pirate_ship.png', help='Input image path')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--ss-steps', type=int, default=12, help='Stage-1 sampling steps')
    parser.add_argument('--ss-cfg-strength', type=float, default=7.5, help='Stage-1 CFG strength')
    parser.add_argument('--slat-steps', type=int, default=12, help='Stage-2 sampling steps')
    parser.add_argument('--slat-cfg-strength', type=float, default=3.0, help='Stage-2 CFG strength')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    # Load an image
    image = Image.open(args.image)

    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_cfg_strength,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_cfg_strength,
        },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
    imageio.mimsave(os.path.join(args.output_dir, "sample_gs.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0], bg_color=(1, 1, 1))['color']
    imageio.mimsave(os.path.join(args.output_dir, "sample_rf.mp4"), video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(os.path.join(args.output_dir, "sample_mesh.mp4"), video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(os.path.join(args.output_dir, "sample.glb"))

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(os.path.join(args.output_dir, "sample.ply"))


if __name__ == '__main__':
    main()
