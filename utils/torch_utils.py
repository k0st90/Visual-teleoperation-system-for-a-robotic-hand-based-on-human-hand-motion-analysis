import torch
from .pytorch3d.rotation_conversions import matrix_to_quaternion  # noqa: F401


def quaternion_angular_error(q1, q2, epsilon=1e-7):
    """
    Angular error between two unit quaternions (w, x, y, z).
    Args: q1, q2 — (..., 4) torch tensors
    Returns: (...,) torch tensor, angle in radians
    """
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    dot = torch.matmul(q1.unsqueeze(-2), q2.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 2.0 * torch.acos(torch.clamp(torch.abs(dot), min=-1.0 + epsilon, max=1.0 - epsilon))
