import argparse
from pathlib import Path
import time
import torch
from mmcv import Config
from mmcls.models import build_classifier


def validate_model_configs(
    config_path="configs/_base_/models/*.py",
    whole_test=False,
    n_batches=1,
    batch_size=1,
    input_size=None,
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

    if config_path.startswith("http"):
        from proc_run.proc_run import proc_run

        config_name = Path(config_path).name
        tmp_path = "/tmp/" + config_name
        proc_run(["wget", config_path, "-O", tmp_path])
        path_ls = [tmp_path]
    else:
        path_ls = sorted([str(p) for p in Path().glob(config_path)])

    for path in path_ls:
        p = Path(path)
        cfg = Config.fromfile(str(p))
        cfg.model.pretrained = None
        model = build_classifier(cfg.model)

        model.eval()

        if input_size is None:
            if hasattr(model.backbone, "timm_model"):
                input_size = model.backbone.timm_model.default_cfg.get(
                    "input_size", (3, 224, 224)
                )
            else:
                input_size = (3, 224, 224)
        elif isinstance(input_size, int):
            input_size = (3, input_size, input_size)

        if max(input_size[1:]) > 456:
            print(f"Big input size {input_size} was clipped.")
            input_size = (input_size[0], 456, 456)

        input_shape = (batch_size, *input_size)

        # print(f"Config: {p.stem: <30} | Input shape: {input_shape}")

        inp = torch.randn(input_shape)

        _time_begin = time.time()
        for _ in range(n_batches):
            if whole_test:
                outputs = model.forward(inp, return_loss=False, img_metas=None)
            else:
                outputs = model.backbone(inp)
        _time_end = time.time()
        _time = _time_end - _time_begin
        _time_per_batch = _time / n_batches

        output_desc = "Outputs" if whole_test else "Backbone Outputs"
        if isinstance(outputs, (tuple, list)):
            output_shapes = [output.shape for output in outputs]
        else:
            output_shapes = outputs.shape
        print(
            f"Config: {p.stem: <30}  | Input: {inp.shape} | {output_desc}: {output_shapes}"
            f" | Batch duration (sec): {_time_per_batch: .6f}"
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
    parser = argparse.ArgumentParser(
        description="Validate mmcls model configs",
    )
    parser.add_argument("config", help="model config file path")
    parser.add_argument(
        "-w",
        "--whole-test",
        action="store_true",
        help="Test the whole model including neck and head",
    )
    parser.add_argument(
        "-n", "--n-iter", type=int, default=1, help="How many times to iterate"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="How many samples in each batch"
    )
    parser.add_argument(
        "-i",
        "--input-size",
        type=int,
        default=None,
        help="Input image width and height",
    )
    parser.add_argument(
        "-o", "--onnx", action="store_true", help="Generate ONNX from Pytorch"
    )
    parser.add_argument(
        "-d", "--dot", action="store_true", help="Generate Dot from ONNX"
    )
    parser.add_argument(
        "-s", "--svg", action="store_true", help="Generate SVG from Dot"
    )
    args = parser.parse_args()

    validate_model_configs(
        args.config,
        args.whole_test,
        args.n_iter,
        args.batch_size,
        args.input_size,
        args.onnx,
        args.dot,
        args.svg,
    )
