# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="TIMMBackbone",
        model_name="swin_large_patch4_window12_384_in22k",
        pretrained=False,
    ),
    head=dict(
        type="LinearClsHead",
        num_classes=10,
        in_channels=1536,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
