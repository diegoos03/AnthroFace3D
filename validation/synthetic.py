import numpy as np
import torch

from core.measurements import (
    nasal_index,
    mouth_measures,
    intercanthal_biocular,
    facial_angles,
    eyebrow_measures,
    facial_index,
    facial_width_ratios,
    facial_asymmetry,
)

# The dimensionless primary measures validated by the synthetic round-trip.
# All are scale-free (ratios or angles), so comparing the true value against
# the recovered one is meaningful even though the 3DMM has no metric unit.
# The first four are pose-invariant on the posed shape (v3d); the eyebrow tilt
# is a frontal-plane angle, so it is measured on the canonical (unposed) shape.
def dimensionless_measures(v3d_np, canonical_np, ldm68_idx):
    brow_tilt = 0.5 * (
        eyebrow_measures(canonical_np, ldm68_idx, "left")["eyebrow_tilt"]
        + eyebrow_measures(canonical_np, ldm68_idx, "right")["eyebrow_tilt"]
    )
    ratios = facial_width_ratios(v3d_np, ldm68_idx)
    asymmetry = facial_asymmetry(v3d_np, canonical_np, ldm68_idx)
    return {
        "nasal_index": nasal_index(v3d_np, ldm68_idx)["nasal_index"],
        "mouth_nose_ratio": mouth_measures(v3d_np, ldm68_idx)["mouth_nose_ratio"],
        "canthal_index": intercanthal_biocular(v3d_np, ldm68_idx)["canthal_index"],
        "nasal_tip_angle": facial_angles(v3d_np, ldm68_idx)["nasal_tip_angle"],
        "eyebrow_tilt": brow_tilt,
        "facial_index": facial_index(v3d_np, ldm68_idx)["facial_index"],
        "naso_intercanthal_ratio": ratios["naso_intercanthal_ratio"],
        "fissure_biocular_ratio": ratios["fissure_biocular_ratio"],
        "mouth_biocular_ratio": ratios["mouth_biocular_ratio"],
        "fissure_asymmetry": asymmetry["fissure_asymmetry"],
        "eyebrow_length_asymmetry": asymmetry["eyebrow_length_asymmetry"],
    }


def base_alpha_from_image(image, recon_model, face_detector, device):
    """
    Regress a real coefficient vector from an image, to use as a realistic
    anchor: synthetic faces are sampled around it, so they stay in-distribution
    (real albedo/lighting/pose, only the identity shape is varied).
    """
    _, im_tensor = face_detector(image)
    recon_model.input_img = im_tensor.to(device)
    with torch.no_grad():
        alpha = recon_model.net_recon(recon_model.input_img)
    return alpha.detach()


def sample_identity(alpha_base, sigma, generator):
    """
    Return a copy of alpha_base with its identity block (first 80 coeffs)
    perturbed by Gaussian noise. Everything else (expression, albedo, lighting,
    pose) is kept from the anchor, so only the face shape -- and thus the
    measures -- varies.
    """
    alpha = alpha_base.clone()
    noise = torch.randn(80, generator=generator, dtype=alpha.dtype) * sigma
    alpha[0, :80] = alpha[0, :80] + noise.to(alpha.device)
    return alpha


def set_pose(alpha, pitch=0.0, yaw=0.0, roll=0.0):
    """
    Return a copy of alpha with its pose block (angles 224:227, in radians)
    overwritten. Order is [pitch(x), yaw(y), roll(z)] per compute_rotation;
    yaw is the left-right turn that drives self-occlusion.
    """
    alpha = alpha.clone()
    alpha[0, 224:227] = torch.tensor([pitch, yaw, roll], dtype=alpha.dtype, device=alpha.device)
    return alpha


def _mesh_and_render(recon_model, alpha):
    """Rebuild the mesh and its shaded render from a coefficient vector,
    replicating the relevant steps of recon_model.forward()."""
    d = recon_model.split_alpha(alpha)
    face_shape = recon_model.compute_shape(d["id"], d["exp"])
    rotation = recon_model.compute_rotation(d["angle"])
    v3d = recon_model.to_camera(recon_model.transform(face_shape, rotation, d["trans"]))

    face_albedo = recon_model.compute_albedo(d["alb"])
    face_norm_roted = recon_model.compute_norm(face_shape) @ rotation
    face_texture = recon_model.compute_texture(face_albedo, face_norm_roted, d["sh"])

    _, _, pred_image, _ = recon_model.renderer(
        v3d.clone(), recon_model.tri, torch.clamp(face_texture, 0, 1).clone(), visible_vertice=True
    )
    render = pred_image.permute(0, 2, 3, 1)[0]  # (H, W, 3) in [0, 1]
    return v3d, face_shape, render


def true_measures(recon_model, alpha, ldm68_idx):
    """Measures computed directly on the known generated mesh (ground truth)."""
    with torch.no_grad():
        v3d, face_shape, _ = _mesh_and_render(recon_model, alpha)
    return dimensionless_measures(
        v3d.detach().cpu().numpy()[0], face_shape.detach().cpu().numpy()[0], ldm68_idx
    )


def recovered_measures(recon_model, alpha, ldm68_idx, device):
    """
    Render the generated mesh to an image, feed it back through the model
    exactly as the pipeline would (net_recon -> mesh), and measure the result.
    The render is already a 224x224 face in the model's frame, so face
    detection and cropping are bypassed (no dependency on RetinaFace liking a
    synthetic render).
    """
    with torch.no_grad():
        _, _, render = _mesh_and_render(recon_model, alpha)
        recon_model.input_img = render.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        results = recon_model.forward()
    return dimensionless_measures(results["v3d"][0], results["face_shape"][0], ldm68_idx)
