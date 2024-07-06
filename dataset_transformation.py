import cv2
import numpy as np
import os
import random
import shutil
from tqdm import tqdm


def create_normal_dataset(input_dir, output_dir, num_images=2000):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(image_files) > num_images:
        selected_files = random.sample(image_files, num_images)
    else:
        print(
            f"Warning: Only {len(image_files)} images found in {input_dir}. Copying all."
        )
        selected_files = image_files

    for filename in tqdm(selected_files, desc=f"Copying {os.path.basename(input_dir)}"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(src_path, dst_path)


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def apply_unsharp_mask(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def apply_subtle_sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    result = cv2.addWeighted(image, 0.7, sharpened, 0.1, 0)
    return result


def transform_image(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe_image = apply_clahe(image)
    unsharp_image = apply_unsharp_mask(image)
    subtle_sharpen = apply_subtle_sharpen(image)

    pseudo_rgb = cv2.merge([clahe_image, unsharp_image, subtle_sharpen])

    return pseudo_rgb


def process_and_save_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        transformed_img = transform_image(img)

        output_filename = os.path.splitext(filename)[0] + "_transformed.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, transformed_img)


# Main execution
if __name__ == "__main__":
    dataset_dir = "./Dataset"
    normal_dataset_dir = "./normal_dataset"
    transformed_dataset_dir = "./pseudo-RGB_dataset"

    categories = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    if len(categories) != 4:
        raise ValueError(f"Expected 4 category folders, but found {len(categories)}")

    print("Creating normal dataset...")
    for category in categories:
        input_category_dir = os.path.join(dataset_dir, category)
        output_category_dir = os.path.join(normal_dataset_dir, category)
        create_normal_dataset(input_category_dir, output_category_dir)

    print("Normal dataset created in:", normal_dataset_dir)

    print("\nProcessing and transforming images...")
    for category in categories:
        input_category_dir = os.path.join(normal_dataset_dir, category)
        output_category_dir = os.path.join(transformed_dataset_dir, category)
        process_and_save_images(input_category_dir, output_category_dir)

    print("Processing complete. Transformed images saved in:", transformed_dataset_dir)
