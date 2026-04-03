# 0) 进入项目
cd /data/kmxu/3D_generation/TRELLIS
conda activate trellispre

# 1) 安装 toolkit 依赖（只需一次）
. ./dataset_toolkits/setup.sh

# 2) 设定输出目录
OUT=/data_nvme7T/kmxu/datasets/ABO

# 3) 初始化 metadata.csv
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 4) 下载 ABO 原始包（若已下载可跳过）
mkdir -p $OUT/raw
aria2c -x 16 -s 16 -k 1M -c \
  -o abo-3dmodels.tar \
  https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar \
  -d $OUT/raw

# 5) 解包+sha256校验（download.py 真正做的是这一步）
python dataset_toolkits/download.py ABO --output_dir $OUT

# 6) 合并下载结果到 metadata
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 7) 渲染多视角（建议 4 卡 + 4 worker）
CUDA_VISIBLE_DEVICES=0,1,2,3 python dataset_toolkits/render.py ABO \
  --output_dir $OUT \
  --num_views 150 \
  --max_workers 4

# 8) 合并渲染结果
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 9) 体素化
python dataset_toolkits/voxelize.py ABO --output_dir $OUT

# 10) 合并体素化结果
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 11) 提取 DINO 特征
python dataset_toolkits/extract_feature.py --output_dir $OUT

# 12) 合并特征结果
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 13) 编码稀疏结构潜变量（ss_latent）
python dataset_toolkits/encode_ss_latent.py --output_dir $OUT

# 14) 合并 ss_latent 结果
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 15) 编码 SLAT 潜变量（latent）
python dataset_toolkits/encode_latent.py --output_dir $OUT

# 16) 合并 latent 结果
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT

# 17) 渲染条件图用于 image-conditioned 训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python dataset_toolkits/render_cond.py ABO \
  --output_dir $OUT \
  --num_views 24 \
  --max_workers 4

# 18) 合并 cond 渲染结果
python dataset_toolkits/build_metadata.py ABO --output_dir $OUT