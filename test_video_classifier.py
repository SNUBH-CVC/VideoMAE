import csv
import logging
import io
import tqdm
import torch
import pydicom
import pandas as pd
import numpy as np
import os
from datetime import datetime
from skimage import exposure, img_as_ubyte
from torchvision.transforms import v2
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv3DSimple, BasicStem


class CSVFormatter(logging.Formatter):

    def __init__(self, header):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)
        self.header = ["time"] + header

    def format(self, record):
        row = []
        for h in self.header:
            if h == "time":
                row.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                continue
            else:
                v = record.msg.get(h, "")
                row.append(v)
        self.writer.writerow(row)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


class CSVRowFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, dict):
            return False
        else:
            return True


class StringFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, dict):
            return True
        else:
            return False


def setup_logger(csv_log_path):
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

    # Create a file handler to log to a file
    csv_file_handler = logging.FileHandler(csv_log_path, encoding="utf-8")
    csv_formatter = CSVFormatter(header=["dcm_path", "video_class"])
    csv_file_handler.setFormatter(csv_formatter)
    csv_file_handler.addFilter(StringFilter())
    logger.addHandler(csv_file_handler)
    return logger


def extract_test_data(prop=None):
    df = pd.read_csv(
        os.path.join("./video_classifier_logs", "240619_video_class_label.csv"),
        index_col=0
    )
    df.patient_id = df.patient_id.astype(str)
    df.study_date = df.study_date.astype(str)

    df = df[df["video_class"].isna()]  # label이 없는 것들만 선택
    
    if prop is not None:  # e.g. prop = 0.001
        test_indices = np.random.choice(
            np.arange(0, len(df)),
            int(len(df) * prop),
            replace=False
        )
        test_df = df.iloc[test_indices]
    else:
        test_df = df

    test_df.to_csv(os.path.join("./video_classifier_logs", "test.csv"))
    print(f"# of test data: {len(test_df)}")
    return test_df


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
        try:
            dcm = pydicom.dcmread(dcm_path)
            pixel_array = dcm.pixel_array
            return dcm, pixel_array
        except Exception:
            print(f"Value error in {row}.")
            return None, None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor([row.video_class])
        
        loaded_video = False
        while not loaded_video:
            dcm, pixel_array = self._read_dcm(row)
            loaded_video = dcm is not None
            if not loaded_video:
                idx = np.random.randint(0, high=len(self) - 1, size=1, dtype=int)[0]
                row = self.df.iloc[idx]
                label = torch.tensor([row.video_class])

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
    device = "cuda:1"

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = f"./video_classifier_logs/{now}_tests_results_log.csv"
    logger = setup_logger(csv_log_path)

    test_df = extract_test_data()
    test_ds = CAGDataset(test_df, mode="test")
    test_data_loader = torch.utils.data.DataLoader(
        test_ds,
        shuffle=False,
        batch_size=1,
        num_workers=1
    )

    model, _, _ = get_model()
    model.load_state_dict(torch.load("./video_classifier_logs/epoch_39.pth"))
    model.to(device)
    model.eval()

    ### test
    dcm_path_list = []
    class_list = []
    with torch.no_grad():
        for data in tqdm.tqdm(test_data_loader):
            inputs, dcm_paths = data
            outputs = model(inputs.to(device))
            outputs = outputs.sigmoid()
            outputs = (outputs > 0.5)

            logger.error({"dcm_path": dcm_paths[0], "video_class": outputs.squeeze().item()})
            dcm_path_list.append(dcm_paths[0])
            class_list.append(outputs.squeeze().item())
    
    result_df = pd.DataFrame({
        "dcm_path": dcm_path_list, 
        "class": class_list
    })
    result_df.to_csv(f"./video_classifier_logs/{now}_test_results.csv")
