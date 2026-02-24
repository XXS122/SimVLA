# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

## Installation

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download LIBERO dataset and place it in `./datasets/metas/`.

### 2. Create Training Metadata

```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### 3. Compute Normalization Statistics

```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### 4. Start Training

**Small Model Configuration:**
```bash
bash train_smolvlm_small.sh
```

**Large Model Configuration:**
```bash
bash train_smolvlm_large.sh
```

### 5. Evaluation

```bash
cd evaluation/libero
```

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads
- **Action Space**: 7-dim (delta xyz + delta euler + gripper)
- **State Space**: 8-dim (ee_pos + axis_angle + gripper_states)

## Reference

If you find our codes useful, please consider citing our work

```
@misc{luo2026simvlasimplevlabaseline,
      title={SimVLA: A Simple VLA Baseline for Robotic Manipulation}, 
      author={Yuankai Luo and Woping Chen and Tong Liang and Baiqiao Wang and Zhenguo Li},
      year={2026},
      eprint={2602.18224},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.18224}, 
}
```


