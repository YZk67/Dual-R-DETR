from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = True  # useful for debugging distributed training

# define dataset for train
coco_path = "../autodl-tmp/COCO2017"  # /PATH/TO/YOUR/COCODIR
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train2017",
    ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=presets.detr,  # see transforms/presets to choose a transform
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val2017",
    ann_file=f"{coco_path}/annotations/instances_val2017.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "configs/deformable_detr_pp/def_detr_pp_resnet_800_1333.py" # def-detr

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
resume_from_checkpoint = None

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

# router gating schedule
gating_lambda_max = 1.0
gating_warmup_epochs = 1
gating_ramp_epochs = 4
gating_last_n_layers = 6
gating_enable_in_eval = True

# aux branch loss weight schedule
aux_alpha_const = None
aux_alpha_min = 0.3
aux_alpha_max = 0.7
aux_alpha_warmup_epochs = gating_warmup_epochs
aux_alpha_ramp_epochs = gating_ramp_epochs
