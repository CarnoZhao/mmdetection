num_classes = 1

img_scale = (800, 800)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3),
    bbox_head=dict(
        type='YOLOXHead', 
        num_classes=num_classes, 
        in_channels=256, 
        feat_channels=256),
    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(
        score_thr=0.01, 
        nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/reef/'
classes = ["starfish"]

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='YOLOMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        size=img_scale,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_base_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomCrop', crop_size=(640,640)),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], flip_ratio=0.5),
    dict(type='AutoAugmentPolicy', autoaug_type="v2"),
]
test_scale=img_scale # (928, 1600)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=test_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                size=test_scale,
                # pad_to_square=False,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
holdout = 0
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
            type='MultiImageMixDataset',
            dataset=[dict(
                        filter_empty_gt=True,
                        classes=classes,
                        type="CocoDataset",
                        ann_file=data_root + f'train/annotations/fold_{fold}.json',
                        img_prefix=data_root + 'train/images/',
                        pipeline=train_base_pipeline) for fold in range(5) if fold != holdout],
            pipeline=train_pipeline),
    val=dict(
            classes=classes,
            type=dataset_type,
                ann_file=data_root + f'train/annotations/fold_{holdout}.json',
                img_prefix=data_root + 'train/images/',
            pipeline=test_pipeline),
    test=dict(
            classes=classes,
            type=dataset_type,
            ann_file=data_root + f'train/annotations/fold_{holdout}.json',
            img_prefix=data_root + 'train/images/',
            pipeline=test_pipeline)
)

nx = 1
work_dir = f'./work_dirs/reef/ylx_l_{nx}x_f{holdout}'
evaluation = dict(
    classwise=True, 
    interval=1, 
    metric='bbox',
    jsonfile_prefix=f"{work_dir}/valid")
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

total_epochs = 20
num_last_epochs = total_epochs // 20
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=False,
    warmup_ratio=1,
    warmup_iters=150,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

checkpoint_config = dict(interval=total_epochs, save_optimizer=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './weights/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=5,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]