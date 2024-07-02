import torch
import pydicom
import pandas as pd
import numpy as np
import os
from skimage import exposure, img_as_ubyte
from torchvision.transforms import v2
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv3DSimple, BasicStem


def split_train_val():
    train_df_path = os.path.join("./video_classifier_logs", "train.csv")
    val_df_path = os.path.join("./video_classifier_logs", "val.csv")
    if os.path.exists(train_df_path):
        train_df = pd.read_csv(train_df_path, index_col=0)
        train_df.patient_id = train_df.patient_id.astype(str)
        train_df.study_date = train_df.study_date.astype(str)

        val_df = pd.read_csv(val_df_path, index_col=0)
        val_df.patient_id = val_df.patient_id.astype(str)
        val_df.study_date = val_df.study_date.astype(str)
    else:
        df = pd.read_csv(
            os.path.join("./video_classifier_logs", "240619_video_class_label.csv"),
            index_col=0
        )
        df.patient_id = df.patient_id.astype(str)
        df.study_date = df.study_date.astype(str)

        df = df[~df["video_class"].isna()]
        df = df[df["video_class"] != 2]

        all_indices = np.arange(0, len(df))
        train_indices = np.random.choice(
            all_indices,
            int(len(df) * 0.9),
            replace=False
        )
        val_indices = np.setdiff1d(all_indices, train_indices)

        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]

        train_df.to_csv(os.path.join("./video_classifier_logs", "train.csv"))
        val_df.to_csv(os.path.join("./video_classifier_logs", "val.csv"))

    print("Train:")
    print("\t 0: ", np.sum(train_df.video_class.values == 0))
    print("\t 1: ", np.sum(train_df.video_class.values == 1))

    print("Val:")
    print("\t 0: ", np.sum(val_df.video_class.values == 0))
    print("\t 1: ", np.sum(val_df.video_class.values == 1))

    return train_df, val_df


def min_max_normalize(pixel_array, return_minmax=False):
    data_min = np.min(pixel_array)
    data_max = np.max(pixel_array)
    normalized = (pixel_array - data_min) / (data_max - data_min)
    if return_minmax:
        return normalized, (data_min, data_max)
    else:
        return normalized


class CAGDataset(torch.utils.data.Dataset):

    def __init__(self, df, mode="train"):
        super(CAGDataset, self).__init__()
        self.root_dir = "/mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data/"
        self.num_retained_frames = 60
        self.df = df
        self.mode = mode
        if self.mode == "train":
            # https://pytorch.org/vision/stable/transforms.html#supported-input-types-and-conventions
            # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
            self.transforms = v2.Compose([
                v2.Resize(size=(256, 256)),
                v2.ElasticTransform(alpha=10.0, sigma=2.0),
                v2.RandomAffine(
                    degrees=(-30, 30),
                    translate=(0, 0.1),
                    scale=(0.9, 1.1)
                ),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:  # val, test
            self.transforms = v2.Compose([
                v2.Resize(size=(256, 256)),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __len__(self):
        return len(self.df)

    def _read_dcm(self, row):
        dcm_path = os.path.join(
            self.root_dir,
            row.patient_id,
            row.study_date,
            row.modality,
            row.file_name
        )
        dcm = pydicom.dcmread(dcm_path)
        pixel_array = dcm.pixel_array
        return dcm, pixel_array

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor([row.video_class])
        try:
            dcm, pixel_array = self._read_dcm(row)
        except (ValueError, AttributeError):
            print(f"Value error in {row}.")
            idx = np.random.randint(0, high=len(self) - 1, size=1, dtype=int)[0]
            row = self.df.iloc[idx]
            label = torch.tensor([row.video_class])
            dcm, pixel_array = self._read_dcm(row)

        # filter out frames
        if dcm.NumberOfFrames < self.num_retained_frames:
            num_padded = self.num_retained_frames - dcm.NumberOfFrames
            num_padded_before = num_padded // 2
            num_padded_after = num_padded - num_padded_before
            _, h, w = dcm.pixel_array.shape
            empty_frame = np.zeros((h, w))
            pixel_array = np.concatenate([
                np.repeat(empty_frame[None], repeats=num_padded_before, axis=0),
                dcm.pixel_array,
                np.repeat(empty_frame[None], repeats=num_padded_after, axis=0)
            ])
        else:
            indices = np.round(
                np.linspace(0, dcm.NumberOfFrames - 1, self.num_retained_frames)
            ).astype(np.int32)
            pixel_array = dcm.pixel_array[indices]

        if pixel_array.dtype == np.uint16:
            pixel_array = img_as_ubyte(exposure.rescale_intensity(pixel_array))

        # min-max normalization
        pixel_array = min_max_normalize(pixel_array)

        # reshape from (T=60, H, W) to (C=1, T=60, H, W)
        pixel_array = pixel_array[None, :, :, :]

        # convert channel from 1 to 3
        pixel_array = torch.from_numpy(pixel_array)
        pixel_array = torch.repeat_interleave(pixel_array, 3, dim=0)  # (C=3, T=60, H, W)

        # apply transform
        pixel_array = self.transforms(pixel_array)
        
        if self.mode == "test":
            dcm_path = os.path.join(
                self.root_dir,
                row.patient_id,
                row.study_date,
                row.modality,
                row.file_name
            )
            return pixel_array, dcm_path
        else:  # train, val
            return pixel_array, label.float()


def get_model():
    model = VideoResNet(
        BasicBlock,
        [Conv3DSimple] * 4,
        [2, 2, 2, 2],
        BasicStem,
        num_classes=1
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return model, optimizer, criterion


if __name__ == "__main__":
    NUM_EPOCHES = 500
    BATCH_SIZE = 8
    device = "cuda"

    train_df, val_df = split_train_val()
    train_ds = CAGDataset(train_df, mode="train")
    val_ds = CAGDataset(val_df, mode="val")

    train_data_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=8
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_ds,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=8
    )

    model, optimizer, criterion = get_model()
    model.to(device)

    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    for epoch in range(NUM_EPOCHES):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs.sigmoid(), labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % 20 == 0:    # print every 5 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

        ### validation
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            model.eval()
            for data in val_data_loader:
                inputs, labels = data
                labels = labels.to(device)
                outputs = model(inputs.to(device))
                outputs = outputs.sigmoid()
                outputs = (outputs > 0.5)
                correct += torch.sum(outputs == labels)
                total += len(outputs)
            print(f"Accuracy: {correct / total:.04f}")

        ### save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                os.path.join("./video_classifier_logs", f"epoch_{epoch}.pth")
            )
        