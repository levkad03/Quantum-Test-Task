import os
import random

from model import LoFTR_Matcher

image_folder_path = "image_dataset"

# Find all the images
image_list = [
    f for f in os.listdir(image_folder_path) if f.endswith((".jpg", ".png", ".jpeg"))
]

# Get 2 random images from folder
img1, img2 = random.sample(image_list, 2)

# Get the path of the image
img1_path = os.path.join(image_folder_path, img1)
img2_path = os.path.join(image_folder_path, img2)

# Define matcher
matcher = LoFTR_Matcher(device="cuda")
img1, img2, mkpts0, mkpts1 = matcher.match_images(img1_path, img2_path)

# Visualize matches
matcher.visualize_matches(img1, img2, mkpts0, mkpts1)
