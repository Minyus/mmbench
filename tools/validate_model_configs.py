from pathlib import Path
import time
import torch
from mmcv import Config
from mmcls.models import build_classifier


def validate_model_configs(
    config_path="configs/_base_/models/*.py",
    n_batches=1,
    batch_size=1,
    to_onnx=False,
    to_dot=False,
    to_svg=False,
):
    if to_onnx:
        import onnx
        from deployment.pytorch2onnx import pytorch2onnx
    if to_dot:
        from deployment.net_drawer import GetPydotGraph, GetOpNodeProducer, OP_STYLE
    if to_svg:
        from proc_run.proc_run import proc_run

    path_ls = sorted([str(p) for p in Path().glob(config_path)])

    for path in path_ls:
        p = Path(path)
        cfg = Config.fromfile(str(p))
        cfg.model.pretrained = None
        model = build_classifier(cfg.model)

        model.eval()

        input_size = model.backbone.timm_model.default_cfg.get(
            "input_size", (3, 224, 224)
        )

        if max(input_size[1:]) > 456:
            print(f"Big input size {input_size} was clipped.")
            input_size = (input_size[0], 456, 456)

        input_shape = (batch_size, *input_size)

        # print(f"Config: {p.stem: <30} | Input shape: {input_shape}")

        inp = torch.randn(input_shape)

        _time_begin = time.time()
        for _ in range(n_batches):
            outputs = model.forward(inp, return_loss=False, img_metas=None)
        _time_end = time.time()
        _time = _time_end - _time_begin
        _time_per_batch = _time / n_batches

        print(
            f"Config: {p.stem: <30} | Time per batch (sec): {_time_per_batch: .6f} | Input shape: {input_shape} | Output shape: {outputs[0].shape}"
        )

        if to_onnx:
            onnx_path = f"onnx/{p.stem}.onnx"
            pytorch2onnx(
                model=model,
                input_shape=input_shape,
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
