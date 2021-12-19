# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="TIMMBackbone", model_name="efficientnetv2_s", pretrained=False),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=10,
        in_channels=1280,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
