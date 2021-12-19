# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="TIMMBackbone",
        model_name="swin_base_patch4_window7_224_in22k",
        pretrained=False,
    ),
    head=dict(
        type="LinearClsHead",
        num_classes=10,
        in_channels=1024,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
