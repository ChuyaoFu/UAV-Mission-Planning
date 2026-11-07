import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from select_estimator_reference import select


class ReferenceSelectionTests(unittest.TestCase):
    def test_robot_transform_is_inverted_once_without_changing_noise(self):
        q = Rotation.from_euler('y', 20, degrees=True).as_quat()
        robot = {'imu_in_body_m': [.22, 0, .055], 'imu_in_body_quaternion_xyzw': q.tolist()}
        text = select('calib_cam_timeoffset: false\nsigma_w: 0.012\n', robot, True)
        self.assertIn('sigma_w: 0.012', text)
        import ast
        transform = np.array(ast.literal_eval(text.split('home_relative_imu_from_body: ')[1]))
        np.testing.assert_allclose(Rotation.from_quat(q).apply(transform[:3]), [-.22, 0, -.055], atol=1e-14)
        np.testing.assert_allclose((Rotation.from_quat(q)*Rotation.from_quat(transform[3:])).as_matrix(), np.eye(3), atol=1e-14)
        disabled = select(text, robot, False)
        self.assertNotIn('home_relative_imu_from_body:', disabled)
        self.assertEqual(disabled.count('retain_home_clone:'), 1)
        self.assertIn('retain_home_clone: false', disabled)

    def test_unknown_or_online_clock_and_nonunit_robot_quaternion_refuse(self):
        robot = {'imu_in_body_m': [0, 0, 0], 'imu_in_body_quaternion_xyzw': [0, 0, 0, 1]}
        for text in ['calib_cam_timeoffset: true\n', 'sigma_w: 0.012\n']:
            with self.assertRaises(ValueError): select(text, robot, True)
        robot['imu_in_body_quaternion_xyzw'][-1] = 2
        with self.assertRaises(ValueError): select('calib_cam_timeoffset: false\n', robot, True)


if __name__ == '__main__': unittest.main()
