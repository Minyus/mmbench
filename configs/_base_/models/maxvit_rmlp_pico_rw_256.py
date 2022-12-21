# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="TIMMBackbone", model_name="maxvit_rmlp_pico_rw_256", pretrained=False
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=10,
        in_channels=256,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
