from skimage import exposure, filters
import random
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates
import glob
import argparse

import os
import numpy as np
from skimage import io, transform, util
from PIL import Image

DEFAULT_SEED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "seed4")
DEFAULT_AUGMENTED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "augmented4"
)


def find_image_pairs(seed_dir):
    """Find all PNG files and their corresponding _seg.npy files in the directory."""
    png_files = glob.glob(os.path.join(seed_dir, "*.png"))
    image_pairs = []

    for png_file in png_files:
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        mask_file = os.path.join(seed_dir, f"{base_name}_seg.npy")

        if os.path.exists(mask_file):
            image_pairs.append((png_file, mask_file, base_name))
        else:
            print(f"Warning: No corresponding mask file found for {png_file}")

    return image_pairs


def load_image_and_mask(image_path, mask_path):
    """Load a single image and mask pair."""
    image = np.array(Image.open(image_path))
    mask_dict = np.load(mask_path, allow_pickle=True).item()
    mask = mask_dict.get("masks")
    mask = np.squeeze(mask)
    if mask.ndim < 2:
        raise ValueError(
            f"Loaded mask shape {mask.shape} is not at least 2D. Check the mask file."
        )
    return image, mask, mask_dict


def save_augmented(
    image, mask, base_name, aug_idx, mask_dict_template, output_dir, orig_image=None
):
    """Save augmented image and mask with unique naming."""
    img_name = f"{base_name}_aug_{aug_idx:03d}.png"
    mask_name = f"{base_name}_aug_{aug_idx:03d}_seg.npy"

    # If image is in [0,1], rescale to match original image's scale
    if image.max() <= 1 and orig_image is not None:
        orig_min, orig_max = orig_image.min(), orig_image.max()
        image = image.astype(np.float32)
        image = image * (orig_max - orig_min) + orig_min
        image = np.clip(image, orig_min, orig_max)
        image = image.astype(np.uint8)
    elif image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    io.imsave(os.path.join(output_dir, img_name), image.astype(np.uint8))
    # Copy the original mask dict and replace only the 'masks' key
    mask_dict = dict(mask_dict_template)
    mask_dict["masks"] = mask
    np.save(os.path.join(output_dir, mask_name), mask_dict, allow_pickle=True)


def augmentations(image, mask):
    aug_pairs = []
    # Original
    aug_pairs.append((image, mask))
    # Horizontal flip
    aug_pairs.append((np.fliplr(image), np.fliplr(mask)))
    # Vertical flip
    aug_pairs.append((np.flipud(image), np.flipud(mask)))
    # 90 degree rotation
    aug_pairs.append((np.rot90(image, 1), np.rot90(mask, 1)))
    # 180 degree rotation
    aug_pairs.append((np.rot90(image, 2), np.rot90(mask, 2)))
    # 270 degree rotation
    aug_pairs.append((np.rot90(image, 3), np.rot90(mask, 3)))
    # Add Gaussian noise (only to image)
    noisy_img = util.random_noise(image, mode="gaussian", var=0.01)
    aug_pairs.append((np.clip(noisy_img * 255, 0, 255).astype(np.uint8), mask))

    # Random brightness adjustment
    bright = exposure.adjust_gamma(image, gamma=random.uniform(0.7, 1.5))
    aug_pairs.append((bright.astype(np.uint8), mask))

    # Random contrast adjustment
    contrast = exposure.rescale_intensity(
        image, in_range=(np.percentile(image, 5), np.percentile(image, 95))
    )
    aug_pairs.append((contrast.astype(np.uint8), mask))

    # Random elastic deformation
    def elastic_transform(img, msk, alpha=34, sigma=4):
        random_state = np.random.RandomState(None)
        shape = img.shape[:2]
        dx = (
            gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect")
            * alpha
        )
        dy = (
            gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect")
            * alpha
        )
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1, 1), (x + dx).reshape(-1, 1)

        def map_img(im):
            if im.ndim == 3:
                return np.stack(
                    [
                        map_coordinates(
                            im[..., c], indices, order=1, mode="reflect"
                        ).reshape(shape)
                        for c in range(im.shape[2])
                    ],
                    axis=-1,
                )
            else:
                return map_coordinates(im, indices, order=1, mode="reflect").reshape(
                    shape
                )

        def map_mask(msk):
            return map_coordinates(msk, indices, order=0, mode="reflect").reshape(shape)

        return map_img(img).astype(np.uint8), map_mask(msk).astype(msk.dtype)

    elastic_img, elastic_mask = elastic_transform(image, mask)
    aug_pairs.append((elastic_img, elastic_mask))

    # Random affine transformation (rotation, translation, shear, scale)
    def random_affine(img, msk):
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-0.1, 0.1)
        tx = random.uniform(-10, 10)
        ty = random.uniform(-10, 10)
        af = transform.AffineTransform(
            scale=(scale, scale),
            rotation=np.deg2rad(angle),
            shear=shear,
            translation=(tx, ty),
        )
        img_t = transform.warp(img, af.inverse, mode="reflect", preserve_range=True)
        msk_t = transform.warp(
            msk, af.inverse, order=0, mode="reflect", preserve_range=True
        )
        return img_t.astype(np.uint8), msk_t.astype(msk.dtype)

    affine_img, affine_mask = random_affine(image, mask)
    aug_pairs.append((affine_img, affine_mask))

    # Add Gaussian noise (only to image)
    noisy_img = util.random_noise(image, mode="gaussian", var=0.01)
    aug_pairs.append((np.clip(noisy_img * 255, 0, 255).astype(np.uint8), mask))

    # Random cutout (erasing)
    def random_cutout(img, msk, size=32):
        h, w = img.shape[:2]
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img2 = img.copy()
        msk2 = msk.copy()
        img2[y : y + size, x : x + size] = 0
        msk2[y : y + size, x : x + size] = 0
        return img2, msk2

    cutout_img, cutout_mask = random_cutout(image, mask)
    aug_pairs.append((cutout_img, cutout_mask))

    # Random Poisson noise (only to image)
    poisson_img = util.random_noise(image, mode="poisson")
    aug_pairs.append((np.clip(poisson_img * 255, 0, 255).astype(np.uint8), mask))

    # Random salt & pepper noise (only to image)
    sp_img = util.random_noise(image, mode="s&p", amount=0.02)
    aug_pairs.append((np.clip(sp_img * 255, 0, 255).astype(np.uint8), mask))

    return aug_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Augment multiple image/mask pairs from a folder"
    )
    parser.add_argument(
        "--seed_dir",
        default=DEFAULT_SEED_DIR,
        help="Directory containing PNG images and their corresponding _seg.npy masks",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_AUGMENTED_DIR,
        help="Directory to save augmented images and masks",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all image pairs in the seed directory
    image_pairs = find_image_pairs(args.seed_dir)

    if not image_pairs:
        print(f"No image pairs found in {args.seed_dir}")
        return

    print(f"Found {len(image_pairs)} image pairs to process")

    total_augmented = 0

    for image_path, mask_path, base_name in image_pairs:
        print(f"Processing {base_name}...")

        # Load the image and mask
        image, mask, mask_dict_template = load_image_and_mask(image_path, mask_path)

        # Generate augmentations
        aug_pairs = augmentations(image, mask)

        # Save all augmented versions
        for aug_idx, (img, msk) in enumerate(aug_pairs):
            save_augmented(
                img,
                msk,
                base_name,
                aug_idx,
                mask_dict_template,
                args.output_dir,
                orig_image=image,
            )

        print(f"  Created {len(aug_pairs)} augmented versions")
        total_augmented += len(aug_pairs)

    print(
        f"Total: Saved {total_augmented} augmented image/mask pairs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
