"""Complex surface / internal-topology structural part generation.

Uses a pretrained multimodal backbone (Step-3.7-Flash on the compute box;
a CLIP proxy locally) as a *condition encoder*, and a 3D voxel geometry
decoder trained with rectified flow matching + multi-objective geometry
losses (SDF L1, occupancy BCE, Chamfer, normal/curvature, topology).

This is deliberately NOT next-token prediction: the decoder outputs an
SDF / occupancy voxel field that is marched into a mesh (STL/OBJ).

Flow-matching primitives (rectified_flow_velocity_loss, logit-normal
timestep sampling, flow batch sampling) are reused from
``tower.train.losses`` so the paradigm matches the rest of the project.
"""

from structgen.config import StructGenConfig  # noqa: F401

__all__ = ["StructGenConfig"]
