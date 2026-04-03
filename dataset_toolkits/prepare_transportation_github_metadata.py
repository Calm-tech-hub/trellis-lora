import argparse
import os

import pandas as pd


def build_metadata(root: str, report_csv: str, source_metadata_csv: str | None = None) -> pd.DataFrame:
    report_path = report_csv if os.path.isabs(report_csv) else os.path.join(root, report_csv)
    metadata = pd.read_csv(report_path)

    metadata = metadata[metadata['status'].isin(['downloaded', 'exists'])].copy()
    metadata['mesh_filename'] = metadata['path'].map(lambda x: os.path.basename(x))
    metadata['local_path'] = metadata['mesh_filename'].map(lambda x: os.path.join('meshes', x))
    metadata['split'] = 'train'
    metadata['category'] = 'transportation'
    metadata['rendered'] = False
    metadata['voxelized'] = False
    metadata['num_voxels'] = 0
    metadata['cond_rendered'] = False

    if source_metadata_csv is not None:
        source_path = source_metadata_csv if os.path.isabs(source_metadata_csv) else os.path.join(root, source_metadata_csv)
        source_metadata = pd.read_csv(source_path)
        merge_columns = [c for c in ['sha256', 'aesthetic_score', 'captions'] if c in source_metadata.columns]
        source_metadata = source_metadata[merge_columns].drop_duplicates(subset=['sha256'])
        metadata = metadata.merge(source_metadata, on='sha256', how='left')

    if 'aesthetic_score' not in metadata.columns:
        metadata['aesthetic_score'] = 10.0
    else:
        metadata['aesthetic_score'] = metadata['aesthetic_score'].fillna(10.0)

    columns = [
        'sha256',
        'split',
        'category',
        'local_path',
        'mesh_filename',
        'file_identifier',
        'status',
        'path',
        'bytes',
        'error',
        'aesthetic_score',
        'captions',
        'rendered',
        'cond_rendered',
        'voxelized',
        'num_voxels',
    ]
    metadata = metadata[columns]
    metadata = metadata.drop_duplicates(subset=['sha256']).sort_values('sha256').reset_index(drop=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='transportation_github dataset root directory')
    parser.add_argument('--report_csv', type=str, default='download_report.csv',
                        help='Path to the download report CSV relative to root')
    parser.add_argument('--source_metadata_csv', type=str, default=None,
                        help='Optional source metadata CSV with captions and aesthetic scores')
    parser.add_argument('--output', type=str, default=None,
                        help='Output metadata path, defaults to <root>/metadata.csv')
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.root, 'metadata.csv')
    metadata = build_metadata(args.root, args.report_csv, args.source_metadata_csv)
    metadata.to_csv(output_path, index=False)
    print(f'Saved metadata with {len(metadata)} instances to {output_path}')
