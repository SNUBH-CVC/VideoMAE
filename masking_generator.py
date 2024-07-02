import numpy as np
import cv2


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])  # shape: (self.num_patches_per_frame,)
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask  # shape: (# patches,), 0과 1로 구성됨


class CAGMaskGenerator:

    def __init__(self, patch_size, input_size, mask_ratio=0.5, threshold1=40, threshold2=80):
        self.patch_size = patch_size
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.mask_ratio = mask_ratio
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        # self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = 30  # batch 단위 학습을 위해서 필요함.
        # self.total_masks = self.frames * self.num_masks_per_frame

    def __call__(self, images):
        # images.shape: (C=1, T, H, W)
        images = images.squeeze()
        center_idx = len(images) // 2
        center_img = images[center_idx]

        # print(f"\nimages.shape={images.shape}")
        # print(f"center_img.shape={center_img.shape}")
        if not isinstance(center_img, np.ndarray):
            center_img = center_img.numpy()
            center_img = np.round(center_img).astype(np.uint8)
        # print(f"(after convert) center_img.shape={center_img.shape}\n")
        
        # import matplotlib.pyplot as plt
        # from datetime import datetime
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(center_img)
        # now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # fig.savefig(f"./{now}_center_img.png")
        # plt.close(fig)

        edges = cv2.Canny(
            image=cv2.GaussianBlur(center_img, (3, 3), 0) , 
            threshold1=self.threshold1, 
            threshold2=self.threshold2
        )  # Canny Edge Detection
        ys, xs = np.where(edges)
        edge_coords = np.stack([xs, ys], axis=-1)
        patch_coords = np.unique(edge_coords // self.patch_size, axis=0)
        patch_indices = np.array([
            self.width * row_patch_idx + col_patch_idx
            for col_patch_idx, row_patch_idx in patch_coords
        ])
        if len(patch_indices) >= self.num_masks_per_frame:
            selected_coords_indices = np.random.choice(
                len(patch_indices),
                self.num_masks_per_frame,
                replace=False
            )
            selected_patch_indices = patch_indices[selected_coords_indices]
        else:
            padded_patch_indices = np.random.choice(
                np.setdiff1d(
                    np.arange(0, self.num_patches_per_frame),
                    patch_indices
                ),
                self.num_masks_per_frame - len(patch_coords),
                replace=False
            )
            selected_patch_indices = np.concatenate([
                patch_indices, padded_patch_indices
            ]).astype(np.int16)
        center_mask = np.zeros(self.num_patches_per_frame)
        center_mask[selected_patch_indices] = 1
        masks = np.tile(center_mask, (self.frames, 1)).flatten()
        return masks