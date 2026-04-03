import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models


torch.set_grad_enabled(False)


def get_voxels(instance, view):
    voxel_path = os.path.join(opt.output_dir, 'renders_cond', instance, f'{view:03d}_voxel.ply')
    position = utils3d.io.read_ply(voxel_path)[0]
    coords = ((torch.tensor(position) + 0.5) * opt.resolution).int().contiguous()
    ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


def get_save_path(latent_name, instance, view):
    return os.path.join(opt.output_dir, 'ss_latents', latent_name, instance, f'{view:03d}.npz')


def count_views(instance):
    transforms_path = os.path.join(opt.output_dir, 'renders_cond', instance, 'transforms.json')
    with open(transforms_path, 'r') as fp:
        transforms = json.load(fp)
    return min(opt.num_views, len(transforms['frames']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=12,
                        help='Number of views per instance to encode')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_views'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}_views'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    os.makedirs(os.path.join(opt.output_dir, 'ss_latents', latent_name), exist_ok=True)

    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')

    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = [item.strip() for item in opt.instances.split(',') if item.strip()]
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['cond_rendered'] == True]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    jobs = []
    for sha256 in metadata['sha256'].values:
        try:
            num_views = count_views(sha256)
        except Exception as e:
            print(f'Error reading transforms for {sha256}: {e}')
            continue
        for view in range(num_views):
            save_path = get_save_path(latent_name, sha256, view)
            if os.path.exists(save_path):
                records.append({'sha256': sha256, 'view': view, f'ss_latent_{latent_name}': True})
                continue
            jobs.append((sha256, view))

    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(job):
                sha256, view = job
                try:
                    ss = get_voxels(sha256, view)[None].float()
                    load_queue.put((sha256, view, ss))
                except Exception as e:
                    print(f'Error loading voxel for {sha256} view {view:03d}: {e}')
            loader_executor.map(loader, jobs)

            def saver(sha256, view, pack):
                save_path = get_save_path(latent_name, sha256, view)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, 'view': view, f'ss_latent_{latent_name}': True})

            for _ in tqdm(range(len(jobs)), desc='Extracting per-view sparse structure latents'):
                sha256, view, ss = load_queue.get()
                ss = ss.cuda().float()
                latent = encoder(ss, sample_posterior=False)
                assert torch.isfinite(latent).all(), 'Non-finite latent'
                pack = {
                    'mean': latent[0].cpu().numpy(),
                }
                saver_executor.submit(saver, sha256, view, pack)

            saver_executor.shutdown(wait=True)
    except Exception:
        print('Error happened during processing.')

    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'ss_latent_{latent_name}_{opt.rank}.csv'), index=False)
