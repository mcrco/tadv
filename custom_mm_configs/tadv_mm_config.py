proj_dir = '/home/marco/dev/tadv/'
config_root_dir = proj_dir + 'BEAR/benchmark/BEAR-Standard/configs/'
tadv_path = config_root_dir + '_base_/models/tadv.py'
default_runtime_path = config_root_dir + '_base_/default_runtime.py'
wandb_runtime_path = config_root_dir + '_base_/wandb_runtime.py'
val_test_runtime = config_root_dir + '_base_/val_test_runtime.py'

_base_ = [
        tadv_path, wandb_runtime_path
]

cfg = {}
cfg["stable_diffusion"] = {
        'use_diffusion': True,
        'freeze_encoder': True,
        'use_diffusion': True,
        'num_train_inference_steps': 10,
        'num_val_inference_steps': 30,
        'model': 'runwayml/stable-diffusion-v1-5'
}
cfg["max_epochs"] = 50

cfg["diffusion_batch_videos"] = 1

model_kwargs = {}
model_kwargs['unet_config'] = {'use_attn': True}  # default for use_attn ends up true
cfg['text_conditioning'] = 'blip'
cfg['blip_caption_path'] = 'captions/mpii_tsh_captions.json'
cfg['use_scaled_encode'] = False

cfg['dreambooth_checkpoint'] = None
cfg['textual_inversion_token_path'] = None
cfg['cross_blip_caption_path'] = None
cfg['dataset_len'] = 1000

class_names = ['stir', 'wash objects', 'cut', 'eat(drink)', 'pour', 'clean']

# model settings
model = dict(
    type='Recognizer3DMeta',
    backbone=dict(
        type='TADPVidMM',
        cfg=cfg,
        class_names = class_names
    ),
    cls_head=dict(
        type='RogerioHeadMM',
        num_classes=6,
        in_channels=717),
    # model training and testing settings
    train_cfg=dict(aux_info=['video_name']),
    test_cfg=dict(average_clips='prob'))


# dataset settings
dataset_type = 'RawframeWithMetaDataset'

# should be defined by training script
data_root = ''
ann_file_train = ''
data_root_val = ''
ann_file_test = ''

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode'),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.8),
    #     random_crop=False,
    #     max_wh_scale_gap=0),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NTCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NTCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NTCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=8,
        workers_per_gpu=4
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05d}.jpg',
        start_index=0,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        filename_tmpl='{:05d}.jpg',
        start_index=0,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        filename_tmpl='{:05d}.jpg',
        start_index=0,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    step=[20, 40],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10)
    
total_epochs = 30

# runtime settings
work_dir = './work_dirs/tadv/mpii_tsh_8frame/'
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbHook', name='mpii-tsh_blip_convtrans')
        # dict(type='TensorboardLoggerHook'),
    ])
checkpoint_config = dict(interval=5)
