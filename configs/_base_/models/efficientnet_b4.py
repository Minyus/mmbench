# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="TIMMBackbone", model_name="efficientnet_b4", pretrained=False),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=10,
        in_channels=1792,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
