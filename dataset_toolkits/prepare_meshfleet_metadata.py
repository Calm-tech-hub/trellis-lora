import argparse
import os

import pandas as pd


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    if 'trellis_aesthetic_score' in metadata.columns:
        metadata['aesthetic_score'] = metadata['trellis_aesthetic_score']
    if 'trellis_caption' in metadata.columns:
        metadata['captions'] = metadata['trellis_caption']
    metadata['local_path'] = metadata['mesh_filename'].map(lambda x: os.path.join('meshes', x))
    return metadata


def build_metadata(root: str, train_csv: str, test_csv: str) -> pd.DataFrame:
    train_path = train_csv if os.path.isabs(train_csv) else os.path.join(root, train_csv)
    test_path = test_csv if os.path.isabs(test_csv) else os.path.join(root, test_csv)

    train_metadata = pd.read_csv(train_path)
    train_metadata['split'] = 'train'
    test_metadata = pd.read_csv(test_path)
    test_metadata['split'] = 'test'

    metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)
    metadata = _normalize_metadata(metadata)
    metadata['rendered'] = False
    metadata['voxelized'] = False
    metadata['num_voxels'] = 0
    metadata['cond_rendered'] = False

    columns = [
        'sha256',
        'split',
        'category',
        'local_path',
        'mesh_filename',
        'fileType',
        'source',
        'license',
        'aesthetic_score',
        'cap3D_caption',
        'captions',
        'caption',
        'refined_3d_prompt',
        'avg_height_of_bbox',
        'normalized_height_of_object',
        'normalized_depth_of_object',
        'normalized_width_of_object',
        'normalized_wheelbase',
        'rendered',
        'cond_rendered',
        'voxelized',
        'num_voxels',
    ]
    columns = [c for c in columns if c in metadata.columns]
    metadata = metadata[columns]
    metadata = metadata.drop_duplicates(subset=['sha256']).sort_values('sha256').reset_index(drop=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='MeshFleet dataset root directory')
    parser.add_argument('--train_csv', type=str, default='meshfleet_train.csv',
                        help='Path to the MeshFleet train CSV relative to root')
    parser.add_argument('--test_csv', type=str, default='meshfleet_test.csv',
                        help='Path to the MeshFleet test CSV relative to root')
    parser.add_argument('--output', type=str, default=None,
                        help='Output metadata path, defaults to <root>/metadata.csv')
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.root, 'metadata.csv')
    metadata = build_metadata(args.root, args.train_csv, args.test_csv)
    metadata.to_csv(output_path, index=False)
    print(f'Saved metadata with {len(metadata)} instances to {output_path}')
