from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state
import torch

xml_path = "../../../../assets/mjcf/keti1/urdf/keti1.xml"
t = SkeletonTree.from_mjcf(xml_path)
zero_pose = SkeletonState.zero_pose(t)

# plot_skeleton_state(zero_pose)

local_rotation = zero_pose.local_rotation.clone()
local_rotation[1] = torch.tensor([0, 1, 0, 0])
new_pose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=t,
            r=local_rotation,
            t=zero_pose.root_translation,
            is_local=True
        )
new_pose.local_rotation

plot_skeleton_state(new_pose)

new_pose.global_rotation
