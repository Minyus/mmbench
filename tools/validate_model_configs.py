from pathlib import Path
import time
import torch
from mmcv import Config
from mmcls.models import CLASSIFIERS


def validate_model_configs(
    config_path="configs/_base_/models/*.py",
    n_batches=1,
    batch_size=1,
    channel_size=3,
    height=224,
    width=224,
):

    for p in Path().glob(config_path):
        cfg = Config.fromfile(str(p))
        model = CLASSIFIERS.build(cfg=dict(cfg["model"]))
        model.eval()

        inp = torch.rand(batch_size, channel_size, height, width)

        _time_begin = time.time()
        for _ in range(n_batches):
            outputs = model.forward(inp, return_loss=False, img_metas=None)
        _time_end = time.time()
        _time = _time_end - _time_begin
        _time_per_batch = _time / n_batches

        print(
            f"Config: {p.stem: <30} | Time per batch (sec): {_time_per_batch: .6f} | Output shape: {outputs[0].shape}"
        )


if __name__ == "__main__":
    validate_model_configs()
