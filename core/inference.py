import torch
from external.util.preprocess import back_resize_ldms


def run_3ddfa(image, face_detector, recon_model, device):
    print("[*] Running 3D face reconstruction...")

    # Detect + preprocess
    trans_params, im_tensor = face_detector(image)

    # Forward pass
    recon_model.input_img = im_tensor.to(device)

    with torch.no_grad():
        results = recon_model.forward()

    print("[+] 3D reconstruction completed")

    return results, trans_params


def project_to_original_image(results, recon_model, trans_params):
    v3d = results['v3d']

    persc_proj = recon_model.persc_proj.cpu().numpy()

    face_proj = v3d @ persc_proj
    face_proj = face_proj[..., :2] / face_proj[..., 2:]

    # flip Y axis
    face_proj[:, :, 1] = 224 - 1 - face_proj[:, :, 1]

    # original size
    face_proj_original = back_resize_ldms(face_proj[0], trans_params)

    return face_proj_original