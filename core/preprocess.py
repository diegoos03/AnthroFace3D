from external.ffhq_crop import FFHQCrop
from PIL import Image


def load_and_crop_image(image_path, config):
    print("[*] Loading and cropping image...")

    ffhqcrop = FFHQCrop(
        quad_scale=1.1,
        dlib_model_path=str(config.assets_dir / "shape_predictor_68_face_landmarks.dat")
    )

    image = ffhqcrop.from_image_path(image_path)

    print("[+] Image successfully preprocessed")
    
    return image