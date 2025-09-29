import torch
import torch.nn.functional as F
from  scipy.spatial.transform import Rotation as R

def extract_PSM_kinematics(data):
    """
    Extracts relevant kinematic variables from the full 76-feature input data tensor,
    focusing on the PSM (Patient Side Manipulator) arms only.

    Args:
        data (np.ndarray): Input data tensor of shape either (n_samples, n_features_original=76) or (n_features_original=76,)

    Returns:
        torch.Tensor: Extracted kinematic variables tensor of shape either (n_samples, n_features=24) or (n_features=24,), depending on the input shape.

    Extracts kinematic variables from the input data tensor. The extracted features are,
    in this order (as the columns of the output tensor):
        - dist (1): distance between the two tooltips
        - angle (1): angle between the two Z axes
        - vel_norm_left (1): norm of the linear velocity of the left tooltip
        - vel_norm_right (1): norm of the linear velocity of the right tooltip
        - gripper_angle_left (1): angle of the left gripper
        - gripper_angle_right (1): angle of the right gripper
        - euler_left (3): euler angles (ZYX) of the left tooltip
        - euler_right (3): euler angles (ZYX) of the right tooltip
        - trans_vel_left (3): linear velocity of the left tooltip, xyz
        - trans_vel_right (3): linear velocity of the right tooltip, xyz
        - rot_vel_left (3): angular velocity of the left tooltip, xyz
        - rot_vel_right (3): angular velocity of the right tooltip, xyz

    The input tensor, data, is expected to have the following structure (from JIGSAWS dataset):
        - cols 1-3    (3) : Master left tooltip xyz                    
        - cols 4-12   (9) : Master left tooltip R    
        - cols 13-15  (3) : Master left tooltip trans_vel x', y', z'   
        - cols 16-18  (3) : Master left tooltip rot_vel                
        - cols 19     (1) : Master left gripper angle                  
        - cols 20-38  (19): Master right
        - cols 39-41  (3) : Slave left tooltip xyz
        - cols 42-50  (9) : Slave left tooltip R
        - cols 51-53  (3) : Slave left tooltip trans_vel x', y', z'   
        - cols 54-56  (3) : Slave left tooltip rot_vel
        - cols 57     (1) : Slave left gripper angle                   
        - cols 58-76  (19): Slave right
        
    """


    # Check input shape
    original_shape = data.shape
    if len(original_shape) == 1:
        # This is a single sample, ensure it's a tensor
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.view(1, -1)
    elif len(original_shape) == 2:
        # This is a batch of samples, ensure it's a tensor
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
    else:
        raise ValueError("Input data must be of shape (n_samples, 76) or (76,)")

    if data.shape[1] != 76:
        raise ValueError("Input data must have 76 features")

    # data is a torch tensor of shape (n_samples, n_features_original)
    features = torch.zeros([data.size(0), 24])
    offset = 19

    # Each sample is a row of the input data torch tensor
    for i in range(data.size(0)):

        sample = data[i,:]

        #### - Extraction of raw kinematics from the input var - ####

        # Positions
        position_left = sample[38:41]
        position_right = sample[38+offset:41+offset]

        # Orientations, from rotational 3x3 matrix to euler zyx angles
        # R is stored in the 1x9 original vector chunck as [row1, row2, row3],
        # as oposed to [col1, col2, col3]
        R_left = R.from_matrix(sample[41:50].view(3, 3).numpy())
        R_right = R.from_matrix(sample[41+offset:50+offset].view(3, 3).numpy())
        euler_left = torch.tensor(R_left.as_euler("zyx"), dtype=torch.float)
        euler_right = torch.tensor(R_right.as_euler("zyx"), dtype=torch.float)
        
        # Linear velocities
        trans_vel_left = sample[50:53]
        trans_vel_right = sample[50+offset:53+offset]

        # Angular velocities
        rot_vel_left = sample[53:56]
        rot_vel_right = sample[53+offset:56+offset]

        # Gripper angles
        gripper_angle_left = sample[56]
        gripper_angle_right = sample[56+offset]

        # Belén's features
        dist = torch.norm(position_right - position_left)
        Z_left = torch.tensor(R_left.as_matrix()[:,2])
        Z_right = torch.tensor(R_right.as_matrix()[:,2])
        angle = torch.acos(torch.dot(Z_left, Z_right))
        vel_norm_left = torch.norm(trans_vel_left)
        vel_norm_right = torch.norm(trans_vel_right)

        # save into features var
        features[i,:] = torch.cat(( dist.view(1),
                                    angle.view(1),
                                    vel_norm_left.view(1),
                                    vel_norm_right.view(1),
                                    gripper_angle_left.view(1),
                                    gripper_angle_right.view(1),
                                    euler_left.view(3),
                                    euler_right.view(3),
                                    trans_vel_left.view(3),
                                    trans_vel_right.view(3),
                                    rot_vel_left.view(3),
                                    rot_vel_right.view(3)
        ), dim=0)

    # If the input was a single sample, return a 1D tensor
    if len(original_shape) == 1:
        features = features.view(-1)


    return features