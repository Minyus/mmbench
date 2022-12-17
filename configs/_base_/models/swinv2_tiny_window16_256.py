# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="TIMMBackbone", model_name="swinv2_tiny_window16_256", pretrained=False
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=1,
        in_channels=768,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
