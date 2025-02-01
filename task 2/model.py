import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import torch
from kornia_moons.viz import draw_LAF_matches


class LoFTR_Matcher:
    def __init__(self, model_type="outdoor", device=None):
        """
        Initializes the LoFTR matcher with a specified model type and device.
        :param model_type: Pretrained model type, default is "outdoor".
        :param device: Device to run the model on (CPU or CUDA).
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = K.feature.LoFTR(pretrained=model_type).to(self.device).eval()
        self.max_image_size = 512  # Maximum image size for processing

    def load_image(self, image_path):
        """
        Loads an image, converts it to grayscale, resizes it, and converts it to a
        tensor.
        :param image_path: Path to the image file.
        :return: Tuple containing the original image and the processed tensor.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        # Resize image
        img = cv2.resize(img, (self.max_image_size, self.max_image_size))
        # Convert to tensor and normalize
        img_tensor = K.image_to_tensor(img, False).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        # Ensure image dimensions do not exceed the max size
        if max(img_tensor.shape[-1], img_tensor.shape[-2]) > self.max_image_size:
            img_tensor = K.geometry.resize(
                img_tensor, self.max_image_size, side="long", interpolation="area"
            )

        return img, img_tensor

    def match_images(self, image_path1, image_path2):
        """
        Matches keypoints between two images using the LoFTR model.
        :param image_path1: Path to the first image.
        :param image_path2: Path to the second image.
        :return: Matched keypoints and images.
        """
        img1, img1_tensor = self.load_image(image_path1)
        img2, img2_tensor = self.load_image(image_path2)

        with torch.inference_mode():
            input_dict = {
                "image0": img1_tensor,
                "image1": img2_tensor,
            }
            # Perform feature matching
            correspondences = self.model(input_dict)

        # Extract keypoints from images
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        return img1, img2, mkpts0, mkpts1

    def visualize_matches(self, img1, img2, mkpts0, mkpts1):
        """
        Visualizes the matched keypoints between two images.
        :param img1: First image.
        :param img2: Second image.
        :param mkpts0: Matched keypoints in the first image.
        :param mkpts1: Matched keypoints in the second image.
        """
        # Compute the Fundamental Matrix and find inliers
        Fm, inliers = cv2.findFundamentalMat(
            mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
        )
        inliers = inliers > 0  # Convert inliers to boolean mask
        # Draw matches
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts0).view(1, -1, 2),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts1).view(1, -1, 2),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1),
            ),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB),
            inliers,
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": None,
                "feature_color": (0.2, 0.5, 1),
                "vertical": False,
            },
        )
        plt.show()
