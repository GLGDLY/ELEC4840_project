# %% [markdown]
# ## Setup imports

# %%
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import torch
import matplotlib.pyplot as plt
import shutil
import os
import glob
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDistanceMetric,
)
from monai.data import MetaTensor
from tqdm import tqdm
from monai.transforms import EnsureTyped
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torch.nn as nn


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


def train():
    data_dir = "./dataset"
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "train", "image", "*.nii.gz"))
    )
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "train", "mask", "*.nii.gz"))
    )
    train_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    val_images = sorted(glob.glob(os.path.join(data_dir, "val", "image", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(data_dir, "val", "mask", "*.nii.gz")))
    val_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(val_images, val_labels)
    ]

    test_images = sorted(glob.glob(os.path.join(data_dir, "test", "image", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(data_dir, "test", "mask", "*.nii.gz")))
    test_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(test_images, test_labels)
    ]

    win_size = (64, 64, 16)
    num_samples = 16

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

    # %%
    import nibabel as nib
    from torch.utils.data import Dataset, DataLoader
    from monai.data import CacheDataset, list_data_collate

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
        num_workers=16,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=16,
        collate_fn=list_data_collate,
    )

    val_ds = CacheDataset(
        data=val_data_dicts, transform=val_transforms, cache_rate=0.5, num_workers=16
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        collate_fn=list_data_collate,
    )

    test_ds = CacheDataset(data=test_data_dicts, transform=val_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        collate_fn=list_data_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    max_epochs = 300
    loss_fns = {
        # "Dice": DiceLoss(to_onehot_y=True, softmax=True, include_background=False),
        # "CE": _CELoss(),
        "Combined_1_1_300_epoch_new": CombinedLoss(weights_dice=1.0, weights_ce=1.0),
        # "Combined_7_3_300_epoch": CombinedLoss(weights_dice=0.7, weights_ce=0.3),
        # "Combined_3_7_300_epoch": CombinedLoss(weights_dice=0.3, weights_ce=0.7),
    }

    for loss_name, loss_fn in loss_fns.items():
        print(f"Training with {loss_name} loss")
        writer = SummaryWriter(
            f"runs/{loss_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        best_metric = -1
        best_metric_epoch = -1
        train_loss_values = []
        train_dice_values = []
        val_loss_values = []
        val_dice_values = []
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            dice_metric.reset()

            for batch_data in tqdm(
                train_loader,
                desc=f"Training Epoch {epoch + 1}/{max_epochs}",
                unit="batch",
            ):
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_outputs = [post_pred(i) for i in decollate_batch(outputs)]
                train_labels = [post_label(i) for i in decollate_batch(labels)]

                dice_metric(y_pred=train_outputs, y=train_labels)
                train_loss += loss.item()

            train_loss /= len(train_loader)
            epoch_dice = dice_metric.aggregate().item()

            train_loss_values.append(train_loss)
            train_dice_values.append(epoch_dice)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Dice/train", epoch_dice, epoch)

            print(
                f"Epoch {epoch + 1}/{max_epochs}, Loss: {train_loss:.4f}, Dice: {epoch_dice:.4f}"
            )

            # Val
            model.eval()
            val_loss = 0.0
            dice_metric.reset()

            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation", unit="batch"):
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    val_outputs = sliding_window_inference(
                        val_images, win_size, num_samples, model
                    )

                    loss = loss_fn(val_outputs, val_labels)

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    val_loss += loss.item()

                val_dice = dice_metric.aggregate().item()
                val_loss /= len(val_loader)
                val_loss_values.append(val_loss)
                val_dice_values.append(val_dice)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Dice/val", val_dice, epoch)

                if val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"best_model_{loss_name}.pth")

                print(f"Validation - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    for loss_name, loss_fn in loss_fns.items():
        print(f"Testing with {loss_name} loss")

        post_pred_save = Compose(
            [
                AsDiscreted(keys="pred", argmax=True),
                EnsureTyped(keys="pred", data_type="tensor"),
            ]
        )

        model.load_state_dict(torch.load(f"best_model_{loss_name}.pth"))
        model.eval()
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
                    output_dir=f"./output/{loss_name}",
                    output_postfix="pred",
                    output_ext=".nii.gz",
                    resample=False,
                    separate_folder=False,
                ),
            ]
        )
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
                    test_image, win_size, num_samples, model
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

        print(f"\nTest Results for {loss_name} loss")
        print(f"Dice Score: {dice_score:.4f}")
        print(f"Jaccard Index: {jaccard_score:.4f}")
        print(f"95% Hausdorff Distance: {hd95_score:.4f}")
        print(f"Average Surface Distance: {asd_score:.4f}")

        with open(f"{loss_name}_test_results.txt", "w") as f:
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
