import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, CAGMaskGenerator
from kinetics import VideoClsDataset, VideoMAE
from cag import CAGVideoMAE
from ssv2 import SSVideoClsDataset
import jepa_transforms

# class DataAugmentationForVideoMAE(object):
#     def __init__(self, args):
#         self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
#         self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
#         normalize = GroupNormalize(self.input_mean, self.input_std)
#         self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
#         self.transform = transforms.Compose([                            
#             self.train_augmentation,
#             Stack(roll=False),
#             ToTorchFormatTensor(div=True),
#             normalize,
#         ])
#         if args.mask_type == 'tube':
#             self.masked_position_generator = TubeMaskingGenerator(
#                 args.window_size, args.mask_ratio
#             )

#     def __call__(self, images):
#         process_data, _ = self.transform(images)
#         return process_data, self.masked_position_generator()

#     def __repr__(self):
#         repr = "(DataAugmentationForVideoMAE,\n"
#         repr += "  transform = %s,\n" % str(self.transform)
#         repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
#         repr += ")"
#         return repr


# def build_pretraining_dataset(args):
#     transform = DataAugmentationForVideoMAE(args)
#     dataset = VideoMAE(
#         root=None,
#         setting=args.data_path,
#         video_ext='mp4',
#         is_color=True,
#         modality='rgb',
#         new_length=args.num_frames,
#         new_step=args.sampling_rate,
#         transform=transform,
#         temporal_jitter=False,
#         video_loader=True,
#         use_decord=True,
#         lazy_init=False)
#     print("Data Aug = %s" % str(transform))
#     return dataset


class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        crop_size=224,
    ):

        """
        random_horizontal_flip=True
        random_resize_aspect_ratio=[0.75, 1.35]                                                                               random_resize_scale=[0.3, 1.0]
        reprob=0.0
        # auto_augment=False
        # motion_shift=False
        crop_size=512
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        """

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.crop_size = crop_size
        self.spatial_transform = jepa_transforms.random_resized_crop

    def __call__(self, buffer):
        # buffer.shape: (T, H, W)
        buffer = torch.tensor(buffer[None], dtype=torch.float32)  # (C=1, T, H, W)
        # buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W
        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = jepa_transforms.horizontal_flip(0.5, buffer)

        return buffer



def min_max_normalize(pixel_array, return_minmax=False):
    data_min = np.min(pixel_array)
    data_max = np.max(pixel_array)
    normalized = (pixel_array - data_min) / (data_max - data_min)
    if return_minmax:
        return normalized, (data_min, data_max)
    else:
        return normalized


class CAGDataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.transform = VideoTransform(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=[0.75, 1.35],
            random_resize_scale=[0.3, 1.0],
            crop_size=512,
        )
        # self.masked_position_generator = TubeMaskingGenerator(
        #     args.window_size, args.mask_ratio
        # )
        self.masked_position_generator = CAGMaskGenerator(
            args.patch_size, args.window_size, 0.3
        )

    def __call__(self, images):
        # images.shape: (T, H, W)
        norm_images, (img_min, img_max) = min_max_normalize(images, return_minmax=True)  # np.ndarray, (T, H, W), 
        process_data = self.transform(norm_images)  # (C=1, T, H, W)
        denorm_images = process_data * (img_max - img_min) + img_min
        
        # import matplotlib.pyplot as plt
        # from datetime import datetime
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(center_img)
        # now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # fig.savefig(f"./{now}_center_img.png")
        # plt.close(fig)

        
        masked_position = self.masked_position_generator(denorm_images)  # shape: (# patches,), 0과 1로 구성됨
        return process_data, masked_position


def build_pretraining_dataset(args):
    transform = CAGDataAugmentationForVideoMAE(args)
    dataset = CAGVideoMAE(
        setting=args.data_path,
        transform=transform
    )
    # print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
