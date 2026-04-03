import argparse
import json
import os
import random

from PIL import Image
from tqdm import tqdm


def scan_dir(cond_dir: str):
    transforms_path = os.path.join(cond_dir, 'transforms.json')
    if not os.path.exists(transforms_path):
        return 'missing_transforms', None

    try:
        with open(transforms_path, 'r') as f:
            frames = json.load(f).get('frames', [])
    except Exception:
        return 'bad_transforms', None

    for frame in frames:
        file_path = frame.get('file_path')
        if file_path is None:
            return 'bad_frame', None

        image_path = os.path.join(cond_dir, file_path)
        if not os.path.exists(image_path):
            return 'missing_png', file_path

        image = Image.open(image_path)
        if 'A' not in image.getbands():
            return 'no_alpha', file_path

        alpha = image.getchannel('A')
        if alpha.getbbox() is None:
            return 'empty_alpha', file_path

    return 'ok', None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--renders_cond_dir', type=str, required=True,
                        help='Path to renders_cond directory')
    parser.add_argument('--sample', type=int, default=None,
                        help='Optional number of directories to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--max_bad_print', type=int, default=50,
                        help='Maximum number of bad examples to print')
    args = parser.parse_args()

    dir_names = sorted([
        d for d in os.listdir(args.renders_cond_dir)
        if os.path.isdir(os.path.join(args.renders_cond_dir, d))
    ])

    if args.sample is not None and args.sample < len(dir_names):
        rng = random.Random(args.seed)
        dir_names = rng.sample(dir_names, args.sample)

    stats = {}
    bad_examples = []
    for name in tqdm(dir_names, desc='Scanning cond renders'):
        status, file_path = scan_dir(os.path.join(args.renders_cond_dir, name))
        stats[status] = stats.get(status, 0) + 1
        if status != 'ok' and len(bad_examples) < args.max_bad_print:
            bad_examples.append((name, status, file_path))

    print('total_dirs', len(dir_names))
    for key in sorted(stats.keys()):
        print(f'{key}: {stats[key]}')

    if bad_examples:
        print('bad_examples')
        for item in bad_examples:
            print(item)


if __name__ == '__main__':
    main()
