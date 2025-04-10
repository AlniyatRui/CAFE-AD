import os
import json
from collections import defaultdict

def get_all_tokens(root):
    tokens = []
    for time_dir in os.listdir(root):
        time_path = os.path.join(root, time_dir)
        if os.path.isdir(time_path):
            for type_dir in os.listdir(time_path):
                type_path = os.path.join(time_path, type_dir)
                if os.path.isdir(type_path):
                    for token in os.listdir(type_path):
                        token_path = os.path.join(type_path, token)
                        if os.path.isdir(token_path):
                            tokens.append((time_dir, type_dir, token))
    return tokens

def main():
    root = "/data/workspace/zhangjunrui/Pluto/Datasets/cache_train_100K"
    all_tokens = get_all_tokens(root)
    total_stats = defaultdict(int)
    for _, type_dir, _ in all_tokens:
        total_stats[type_dir] += 1
    type2id = {type_name: idx for idx, type_name in enumerate((total_stats.keys()))}
    avg_count = sum(total_stats.values()) / len(total_stats)
    above_avg_types = [type2id[type_name] for type_name, count in total_stats.items() if count > avg_count]

    with open("preprocess/type2id.json", "w") as f:
        json.dump(type2id, f, indent=2)
    with open("preprocess/above_avg_types.json", "w") as f:
        json.dump(above_avg_types, f, indent=2)

if __name__ == "__main__":
    main()
