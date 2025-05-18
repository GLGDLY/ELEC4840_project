# %% [markdown]
# ## Setup imports

import glob
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    MetaTensor,
    decollate_batch,
    list_data_collate,
)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDistanceMetric,
)
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandRotated,
    RandSpatialCropd,
    RandZoomd,
    SaveImaged,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    KeepLargestConnectedComponent,
)
from monai.utils import first, set_determinism
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from teacher_student import MeanTeacherFramework

import warnings

warnings.filterwarnings("ignore")


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.PReLU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.PReLU(),
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(1, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)
        self.dconv_down5 = double_conv(128, 256)

        self.maxpool = nn.MaxPool3d(2)

        self.upsample4 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dconv_up4 = double_conv(128 + 256, 128)

        self.upsample3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dconv_up3 = double_conv(64 + 128, 64)

        self.upsample2 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.dconv_up2 = double_conv(32 + 64, 32)

        self.upsample1 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.dconv_up1 = double_conv(16 + 32, 16)

        self.conv_last = nn.Conv3d(16, 2, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


class _CELoss(torch.nn.CrossEntropyLoss):
    def forward(self, pred, target):
        return super().forward(pred, target.squeeze(1).long())


class CombinedLoss(torch.nn.Module):
    def __init__(self, weights_dice=0.5, weights_ce=0.5):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.ce = _CELoss()
        self.weights_dice = weights_dice
        self.weights_ce = weights_ce

    def forward(self, pred, target):
        return (
            self.dice(pred, target) * self.weights_dice
            + self.ce(pred, target) * self.weights_ce
        )


################### Config #######################

win_size = (64, 64, 16)
num_samples = 16

train_batch = 8
pred_batch = 1
num_workers = 0

max_epochs = 300
max_semi_epochs = 8
max_semi_training_epochs = 15

loss_name = "Combined_1_1_300_epoch_new"
loss_fn = CombinedLoss(weights_dice=1, weights_ce=1)
consistency_criterion = nn.MSELoss()
consistency_weight = 0.1
consistency_weight_fn = lambda semi_epoch: max(
    0.5, consistency_weight + 0.1 * semi_epoch
)
lr_decay = 1e-4

data_dir = "./dataset"
data_dir_unlabeled = "./dataset_unlabeled"  # "/root/autodl-tmp/dataset_unlabeled"

##################################################


def train():
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "train", "image", "*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "train", "mask", "*.nii.gz"))
    )
    train_data_dicts = [
        {"image": image_name, "label": label_name, "is_pseudo": False}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    val_images = sorted(glob.glob(os.path.join(data_dir, "val", "image", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(data_dir, "val", "mask", "*.nii.gz")))
    val_data_dicts = [
        {"image": image_name, "label": label_name, "is_pseudo": False}
        for image_name, label_name in zip(val_images, val_labels)
    ]

    test_images = sorted(glob.glob(os.path.join(data_dir, "test", "image", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(data_dir, "test", "mask", "*.nii.gz")))
    test_data_dicts = [
        {"image": image_name, "label": label_name, "is_pseudo": False}
        for image_name, label_name in zip(test_images, test_labels)
    ]

    unlabeled_images = sorted(glob.glob(os.path.join(data_dir_unlabeled, "*.nii.gz")))
    unlabeled_data_dicts = [
        {"image": img, "image_str": img} for img in unlabeled_images
    ]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandRotated(
                keys=["image", "label"], prob=0.5, range_x=0.3, range_y=0.3, range_z=0.3
            ),
            RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2),
            RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.05),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.3)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=win_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )

    unlabeled_train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            RandRotated(
                keys=["image"], prob=0.8, range_x=0.4, range_y=0.4, range_z=0.4
            ),
            RandZoomd(keys=["image"], prob=0.8, min_zoom=0.7, max_zoom=1.3),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),
            RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.5, 1.5)),
            Orientationd(keys=["image"], axcodes="RAS"),
        ]
    )

    mixed_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=win_size,
                num_samples=num_samples,
                pos=1,
                neg=1,
            ),
        ]
    )

    # class CT_Dataset(Dataset):
    #     def __init__(self, dataset_path, transform=None,split='test'):
    #         self.dataset_path = dataset_path
    #         self.transform = transform
    #         self.split = split

    #     def __len__(self):
    #         return len(self.dataset_path)

    #     def __getitem__(self, idx):
    #         data = self.dataset_path[idx]
    #         image = nib.load(data['image'])
    #         label = nib.load(data['label'])
    #         image = image.get_fdata()
    #         label = label.get_fdata()
    #         if self.transform:
    #             image, label = self.transform(image, label)
    #         return image, label

    # train_files = train_data_dicts
    # val_files = val_data_dicts
    # test_files = test_data_dicts
    # test_transforms = val_transforms

    # # here we don't cache any data in case out of memory issue
    # train_ds = CT_Dataset(train_files,train_transforms,split='train')
    # train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    # val_ds = CT_Dataset(val_files,val_transforms,split='val')
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    # test_ds = CT_Dataset(test_files,test_transforms,split='test')
    # test_loader = DataLoader(test_ds, batch_size=2, shuffle=True, num_workers=4)
    # val_ds = CT_Dataset(val_files,val_transforms,split='val')
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_rate=0.5,
        num_workers=num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )

    val_ds = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        cache_rate=0.5,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=pred_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )

    test_ds = Dataset(data=test_data_dicts, transform=val_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=pred_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )

    unlabeled_ds = Dataset(
        data=unlabeled_data_dicts,
        transform=unlabeled_train_transforms,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=pred_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )

    class ProgressiveUnfreezer:
        def __init__(self, model):
            self.model = model
            self.stage_layers = {
                0: ["conv_last"],
                1: ["dconv_up1", "dconv_up2"],
                2: ["dconv_up3", "dconv_up4"],
                3: ["dconv_down5", "upsample4"],
                4: ["dconv_down4", "upsample3"],
                5: ["dconv_down3", "upsample2"],
                6: ["dconv_down2", "upsample1"],
                7: ["dconv_down1"],
            }

        def set_stage(self, stage):
            for param in self.model.parameters():
                param.requires_grad = False

            for s in range(stage + 1):
                if s in self.stage_layers:
                    for name, param in self.model.named_parameters():
                        if any(layer in name for layer in self.stage_layers[s]):
                            param.requires_grad = True

    def get_optimizer(model, current_stage):
        lr_config = {
            0: {"lr": 1e-4, "layers": ["conv_last"]},
            1: {"lr": 5e-4, "layers": ["dconv_up1", "dconv_up2"]},
            2: {"lr": 1e-3, "layers": ["dconv_up3", "dconv_up4"]},
            3: {"lr": 2e-3, "layers": ["dconv_down5", "upsample4"]},
            4: {"lr": 3e-3, "layers": ["dconv_down4", "upsample3"]},
            5: {"lr": 3e-3, "layers": ["dconv_down3", "upsample2"]},
            6: {"lr": 3e-3, "layers": ["dconv_down2", "upsample1"]},
            7: {"lr": 3e-3, "layers": ["dconv_down1"]},
        }

        param_groups = []
        for s in range(current_stage + 1):
            if s in lr_config:
                group = {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(l in n for l in lr_config[s]["layers"])
                    ],
                    "lr": lr_config[s]["lr"],
                }
                param_groups.append(group)

        return torch.optim.AdamW(param_groups, weight_decay=lr_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU")
    base_model = UNet().to(device)
    base_model.load_state_dict(torch.load(f"best_model_{loss_name}.pth"))
    model = MeanTeacherFramework(base_model).to(device)
    unfreezer = ProgressiveUnfreezer(model)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Some other transforms
    post_pred_save = Compose(
        [
            AsDiscreted(keys="pred", argmax=True),
            EnsureTyped(keys="pred", data_type="tensor"),
        ]
    )

    post_trans = Compose(
        [
            EnsureTyped(keys="pred", data_type="tensor"),
            Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            SaveImaged(
                keys="pred",
                meta_keys="image_meta_dict",
                output_dir=f"./output/semi_{loss_name}",
                output_postfix="pred",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,
            ),
        ]
    )
    pesudo_post_trans = Compose(
        [
            EnsureTyped(keys="pred", data_type="tensor"),
            Invertd(
                keys="pred",
                transform=unlabeled_train_transforms,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            SaveImaged(
                keys="pred",
                meta_keys="image_meta_dict",
                output_dir=f"./temp_pseudo_labels",
                output_postfix="pred",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,
            ),
        ]
    )

    post_pred = Compose(
        [
            AsDiscrete(argmax=True, to_onehot=2),
        ]
    )
    post_label = Compose(
        [
            AsDiscrete(to_onehot=2),
        ]
    )

    # %% [markdown]
    # ## Define your training/val/test loop

    # %%

    if not os.path.exists("./temp_pseudo_labels"):
        os.makedirs("./temp_pseudo_labels")
    else:
        shutil.rmtree("./temp_pseudo_labels")
        os.makedirs("./temp_pseudo_labels")

    best_metric = -1
    best_metric_semi_epoch = -1
    best_metric_epoch = -1
    for semi_epoch in range(max_semi_epochs):
        print(f"\nGenerating pseudo labels - Iteration {semi_epoch+1}")
        unfreezer.set_stage(semi_epoch)
        optimizer = get_optimizer(model.student, semi_epoch)
        writer = SummaryWriter(
            f"runs/semisupervised_{loss_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_semi_epoch_{semi_epoch+1}"
        )

        this_best_metric = -1
        this_best_metric_epoch = -1

        model.teacher.eval()
        pseudo_label_dicts = []

        confidence_threshold = min(0.9, 0.7 + semi_epoch * 0.1)
        presudo_post_pred_save = Compose(
            [
                AsDiscreted(keys="pred", argmax=True, threshold=confidence_threshold),
                EnsureTyped(keys="pred", data_type="tensor"),
            ]
        )

        with torch.no_grad():
            for data in tqdm(
                unlabeled_loader, desc="Generating pseudo labels", unit="batch"
            ):
                image = data["image"].to(device)

                pred = sliding_window_inference(
                    image, win_size, num_samples, model.teacher
                )

                pseudo_label_save = [
                    presudo_post_pred_save({"pred": i}) for i in decollate_batch(pred)
                ]
                for i in range(len(pseudo_label_save)):
                    # if very few valid labels, skip
                    if pseudo_label_save[i]["pred"].sum() <= 10000:
                        continue
                    sample_data = {
                        "image": data["image"][i],
                        "pred": MetaTensor(
                            pseudo_label_save[i]["pred"],
                            meta=data["image"][i].meta,
                        ),
                    }
                    pesudo_post_trans(sample_data)
                    name = os.path.basename(data["image_str"][i])
                    name = name.replace(".nii.gz", "_pred.nii.gz")
                    pseudo_label_dicts.append(
                        {
                            "image": data["image_str"][i],
                            "label": os.path.join(
                                "./temp_pseudo_labels",
                                name,
                            ),
                            "is_pseudo": True,
                        }
                    )

        combined_dicts = train_data_dicts + pseudo_label_dicts
        print(
            f"Combined dataset size: {len(combined_dicts)} (Labeled: {len(train_data_dicts)}, Pseudo: {len(pseudo_label_dicts)})"
        )

        mixed_ds = Dataset(data=combined_dicts, transform=mixed_transforms)
        mixed_loader = DataLoader(
            mixed_ds,
            batch_size=train_batch,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=list_data_collate,
        )

        final_loss = 0
        final_dice = 0
        for epoch in range(max_semi_training_epochs):
            model.train()
            model.student.train()
            epoch_loss = 0
            dice_metric.reset()

            for batch in tqdm(mixed_loader):
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                with torch.no_grad():
                    teacher_outputs = sliding_window_inference(
                        inputs, win_size, num_samples, model.teacher
                    )

                optimizer.zero_grad()
                student_outputs = model.student(inputs)

                loss_weight = torch.tensor(
                    [
                        1.0 if not batch["is_pseudo"][i] else 0.3
                        for i in range(len(inputs))
                    ]
                ).to(device)

                consistency_loss = consistency_weight_fn(
                    semi_epoch
                ) * consistency_criterion(student_outputs, teacher_outputs)

                loss = 0
                for i in range(len(inputs)):
                    sample_loss = loss_fn(student_outputs[i : i + 1], labels[i : i + 1])
                    weighted_loss = sample_loss * loss_weight[i]
                    loss += weighted_loss

                loss = loss / len(inputs) + consistency_loss
                loss.backward()
                optimizer.step()

                train_outputs = [post_pred(i) for i in decollate_batch(student_outputs)]
                train_labels = [post_label(i) for i in decollate_batch(labels)]

                dice_metric(y_pred=train_outputs, y=train_labels)
                epoch_loss += loss.item()

            # update the teacher model
            model.update_teacher()

            final_loss = epoch_loss / len(mixed_loader)
            final_dice = dice_metric.aggregate().item()
            writer.add_scalar(
                "Loss/train", final_loss, semi_epoch * max_semi_training_epochs + epoch
            )
            writer.add_scalar(
                "Dice/train", final_dice, semi_epoch * max_semi_training_epochs + epoch
            )

            print(
                f"Semi Epoch [{semi_epoch+1}/3] | Epoch [{epoch+1}/30] Loss: {final_loss:.4f} Dice: {final_dice:.4f}"
            )

            # Val
            model.eval()
            model.student.eval()
            val_loss = 0.0
            dice_metric.reset()

            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation", unit="batch"):
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    val_outputs = sliding_window_inference(
                        val_images, win_size, num_samples, model.student
                    )

                    loss = loss_fn(val_outputs, val_labels)

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    val_loss += loss.item()

                val_dice = dice_metric.aggregate().item()
                val_loss /= len(val_loader)
                writer.add_scalar(
                    "Loss/val", val_loss, semi_epoch * max_semi_training_epochs + epoch
                )
                writer.add_scalar(
                    "Dice/val", val_dice, semi_epoch * max_semi_training_epochs + epoch
                )

                if val_dice > this_best_metric:
                    this_best_metric = val_dice
                    this_best_metric_epoch = epoch + 1
                    torch.save(
                        model.student.state_dict(),
                        f"best_model_semisupervised_{loss_name}_semi{semi_epoch}.pth",
                    )

                if val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_semi_epoch = semi_epoch + 1
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.student.state_dict(),
                        f"best_model_semisupervised_{loss_name}.pth",
                    )

                print(f"Validation - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        print(f"Training - Loss: {final_loss:.4f}, Dice: {final_dice:.4f}")
        print(
            f"This best dice: {this_best_metric:.4f} at epoch {this_best_metric_epoch} for semi-supervised {loss_name} loss"
        )
        print(
            f"Best dice: {best_metric:.4f} at semi_epoch {best_metric_semi_epoch}, epoch {best_metric_epoch} for semi-supervised {loss_name} loss"
        )
        # restore best model
        model.student.load_state_dict(
            torch.load(f"best_model_semisupervised_{loss_name}.pth")
        )

    # %% [markdown]
    # ## Inference and Report performance on Test Set

    # %%
    print(f"Testing with {loss_name} loss")

    model.student.load_state_dict(
        torch.load(f"best_model_semisupervised_{loss_name}.pth")
    )
    model.eval()
    model.student.eval()
    os.makedirs(f"./output/{loss_name}", exist_ok=True)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    jaccard_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=False)
    asd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True)

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Testing", unit="batch"):
            test_image = test_data["image"].to(device)
            test_label = test_data["label"].to(device)

            test_output = sliding_window_inference(
                test_image, win_size, num_samples, model.student
            )

            test_output_save = [
                post_pred_save({"pred": i}) for i in decollate_batch(test_output)
            ]
            test_output = [post_pred(i) for i in decollate_batch(test_output)]
            test_labels = [post_label(i) for i in decollate_batch(test_label)]

            dice_metric(y_pred=test_output, y=test_labels)
            jaccard_metric(y_pred=test_output, y=test_labels)
            hd95_metric(y_pred=test_output, y=test_labels)
            asd_metric(y_pred=test_output, y=test_labels)

            for i in range(len(test_output_save)):
                sample_data = {
                    "image": test_data["image"][i],
                    "pred": MetaTensor(
                        test_output_save[i]["pred"], meta=test_data["image"][i].meta
                    ),
                }
                post_trans(sample_data)

    dice_score = dice_metric.aggregate().item()
    jaccard_score = jaccard_metric.aggregate().item()
    hd95_score = hd95_metric.aggregate().item()
    asd_score = asd_metric.aggregate().item()

    print(f"\nTest Results for semi-supervised {loss_name} loss")
    print(f"Dice Score: {dice_score:.4f}")
    print(f"Jaccard Index: {jaccard_score:.4f}")
    print(f"95% Hausdorff Distance: {hd95_score:.4f}")
    print(f"Average Surface Distance: {asd_score:.4f}")

    with open(f"semisupervised_{loss_name}_test_results.txt", "w") as f:
        f.write(f"Test Results for {loss_name} loss\n")
        f.write(f"Dice Score: {dice_score:.4f}\n")
        f.write(f"Jaccard Index: {jaccard_score:.4f}\n")
        f.write(f"95% Hausdorff Distance: {hd95_score:.4f}\n")
        f.write(f"Average Surface Distance: {asd_score:.4f}\n")

    # just in case I want to run again and forgot the reset the things
    dice_metric.reset()
    jaccard_metric.reset()
    hd95_metric.reset()
    asd_metric.reset()


if __name__ == "__main__":
    train()
    shutil.rmtree("./temp_pseudo_labels")
    # os.system("/usr/bin/shutdown")
