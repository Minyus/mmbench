# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(type="TIMMBackbone", model_name="efficientnet_b3", pretrained=False),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=10,
        in_channels=1536,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
