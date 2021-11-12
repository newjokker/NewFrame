_base_ = [
    # '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
    '../_base_/models/upernet_swin.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        # num_classes=150
        num_classes=2  # ---------------------------------------------------------------------------
        # num_classes=3  # ---------------------------------------------------------------------------

    ),
    auxiliary_head=dict(
        in_channels=384,
        # num_classes=150
        num_classes=2  # --------------------------------------------------------------------------
        # num_classes=3  # --------------------------------------------------------------------------

    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone

# By default, models are trained on 8 GPUs with 2 images per GPU
