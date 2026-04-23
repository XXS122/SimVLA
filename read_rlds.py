import tensorflow_datasets as tfds
import tensorflow as tf
import os

# 1. 环境配置：强制使用 CPU 避免抢占 A100/4090 显存，同时屏蔽 TF 冗余日志
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 2. 你的准确路径
DATA_DIR = "/root/VLABench-data/rlds/1.0.0"

def extract_one_frame():
    print(f"正在从目录加载数据集: {DATA_DIR}...")
    
    try:
        # 核心：直接指向包含 dataset_info.json 的文件夹
        builder = tfds.builder_from_directory(DATA_DIR)
        
        # VLABench 通常包含 'train' 或 'test'，如果不确定可以用 builder.info.splits 打印看看
        # 这里尝试加载第一个可用的 split
        split_name = list(builder.info.splits.keys())[0]
        dataset = builder.as_dataset(split=split_name)
        
        print(f"成功加载 Split: {split_name}")
        print("-" * 30)

        # 3. 提取数据
        for episode in dataset.take(1):
            steps = episode['steps']
            for step in steps.take(1):
                # 将 TensorFlow Tensor 转换为 Numpy 数组，方便开发调试
                frame = tfds.as_numpy(step)
                
                # --- 格式化输出 ---
                print("【文本指令】:", frame.get('language_instruction', b'N/A').decode('utf-8'))
                
                print("\n【Observation 状态】:")
                obs = frame.get('observation', {})
                for k, v in obs.items():
                    if hasattr(v, 'shape'):
                        print(f" - {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f" - {k}: {v}")
                
                print("\n【Action 动作】:")
                action = frame.get('action')
                print(f" - action: shape={action.shape if hasattr(action, 'shape') else 'N/A'}, value={action}")
                
                print("\n【其他信息】:")
                print(f" - is_first: {frame.get('is_first')}")
                print(f" - is_last: {frame.get('is_last')}")
                print(f" - reward: {frame.get('reward')}")
                
                return frame

    except Exception as e:
        print(f"解析失败！错误详情:\n{str(e)}")

if __name__ == "__main__":
    frame_data = extract_one_frame()