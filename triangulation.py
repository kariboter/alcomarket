import numpy as np


def find_depth(right_point, left_point, frame_right, frame_left, baseline, alpha):
    f_pixel = 0
    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point
    x_left = left_point

    # CALCULATE THE DISPARITY:
    disparity = x_left - x_right
    # Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    depth = (baseline * f_pixel) / disparity  # Depth in [cm]

    return depth