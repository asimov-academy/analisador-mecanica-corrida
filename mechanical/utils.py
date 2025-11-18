import numpy as np

def calculate_angle(point_x, point_y, zero_line_x, reference_y):
    delta_x = point_x - zero_line_x
    delta_y = reference_y - point_y  # delta_y positivo quando o ponto está acima da referência

    angle_rad = np.arctan2(delta_x, delta_y)
    angle_deg = np.degrees(angle_rad)

    return angle_deg
