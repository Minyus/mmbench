from pathlib import Path
import time
import torch
from mmcv import Config
from mmcls.models import build_classifier

import onnx
from deployment.pytorch2onnx import pytorch2onnx
from deployment.net_drawer import GetPydotGraph, GetOpNodeProducer, OP_STYLE
from proc_run.proc_run import proc_run


def validate_model_configs(
    config_path="configs/_base_/models/*.py",
    n_batches=1,
    batch_size=1,
    channel_size=3,
    height=224,
    width=224,
    to_onnx=True,
    to_dot=True,
    to_svg=True,
):

    for p in Path().glob(config_path):
        print(p)
        cfg = Config.fromfile(str(p))
        cfg.model.pretrained = None
        model = build_classifier(cfg.model)

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

        if to_onnx:
            onnx_path = f"onnx/{p.stem}.onnx"
            pytorch2onnx(
                model=model,
                input_shape=(batch_size, channel_size, height, width),
                opset_version=11,
                dynamic_export=False,
                show=False,
                output_file=onnx_path,
                do_simplify=False,
                verify=False,
            )

            if to_dot:
                dot_path = f"onnx/{p.stem}.dot"
                onnx_msg = onnx.load(onnx_path)
                pydot_graph = GetPydotGraph(
                    onnx_msg.graph,
                    name=onnx_msg.graph.name,
                    rankdir="TB",  # "TB": Top to Bottom, "LR": Left to Right
                    node_producer=GetOpNodeProducer(embed_docstring=True, **OP_STYLE),
                )
                pydot_graph.write_dot(dot_path)
                print(f"Successfully exported DOT: {dot_path}")

                if to_svg:
                    svg_path = f"onnx/{p.stem}.svg"
                    proc_run(["dot", "-Tsvg", dot_path, "-o", svg_path])


if __name__ == "__main__":
    validate_model_configs()
