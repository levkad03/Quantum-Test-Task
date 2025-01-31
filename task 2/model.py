import cv2
import kornia as K
import torch


class LoFTR_Matcher:
    def __init__(self, model_type="outdoor", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = K.feature.LoFTR(pretrained=model_type).to(self.device).eval()

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = K.image_to_tensor(img, False).float() / 255.0

        return img, img_tensor

    def match_images(self, image_path1, image_path2):
        img1, img1_tensor = self.load_image(image_path1)
        img2, img2_tensor = self.load_image(image_path2)

        with torch.no_grad():
            input_dict = {
                "image0": img1_tensor,
                "image1": img2_tensor,
            }
            correspondences = self.model(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        return img1, img2, mkpts0, mkpts1
