import functools
import json
import os
from pathlib import Path
from typing import Dict, List

from safetensors import safe_open


class LazyShardLoader:

    def __init__(self, checkpoint_dir: str, max_open: int = 32, device="cpu"):
        self.ckpt_dir = Path(checkpoint_dir)
        self.device   = device
        self.weight_to_file: Dict[str, str] = self._build_weight_map()

        @functools.lru_cache(maxsize=max_open)
        def _open(fname: str):
            return safe_open(self.ckpt_dir / fname,
                             framework="pt",
                             device=self.device)
        self._open = _open

    def get_tensor(self, name: str):
        shard = self.weight_to_file[name]
        return self._open(shard).get_tensor(name) 

    def close_all(self):
        for h in list(self._open.cache.values()):
            h.close()
        self._open.cache_clear()

    def _build_weight_map(self) -> Dict[str, str]:
        idx_path = self.ckpt_dir / "model.safetensors.index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                return json.load(f)["weight_map"]

        shards: List[Path] = sorted(self.ckpt_dir.glob("*.safetensors"))

        if not shards:
            raise FileNotFoundError("No *.safetensors files found in "
                                    f"{self.ckpt_dir}")

        if len(shards) == 1:
            one = shards[0].name
            with safe_open(shards[0], framework="pt") as f:
                return {k: one for k in f.keys()}

        weight_map: Dict[str, str] = {}
        for shard_path in shards:
            with safe_open(shard_path, framework="pt") as f:
                # f.keys() is cheap – reads the ~1 KB header only
                for k in f.keys():
                    weight_map[k] = shard_path.name
        return weight_map

if __name__ == "__main__":
    import argparse
    from pprint import pprint

    from huggingface_hub import snapshot_download
    from transformers.utils.hub import cached_file
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    chkpt_dir = snapshot_download(args.model_path)
    print(list(Path(chkpt_dir).iterdir()))
    breakpoint()
    loader = LazyShardLoader(chkpt_dir)
    pprint(loader.weight_to_file)