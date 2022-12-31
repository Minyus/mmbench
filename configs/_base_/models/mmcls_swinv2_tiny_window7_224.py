# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="SwinTransformerV2",
        arch="tiny",
        window_size=7,
        img_size=224,
        drop_path_rate=0.2,
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=768,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
