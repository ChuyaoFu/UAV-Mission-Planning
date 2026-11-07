"""Rigid mounting is applied before rendering; native epipolar geometry stays fixed."""
import json,subprocess,sys,tempfile,unittest,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from camera_geometry import load_rig,bearing_visible

ROOT=Path(__file__).resolve().parents[1]


class RigProfiles(unittest.TestCase):
    def test_portrait_and_landscape_preserve_stereo_and_body_fixed_tof(self):
        K,T,baseline,_=load_rig()
        for roll,pitch in [(0,0),(0,15),(90,5),(-90,5)]:
            with self.subTest(roll=roll,pitch=pitch),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                for name in ['model.sdf','robot.json','navigation.json']:
                    shutil.copy2(ROOT/'config'/name,folder/name)
                subprocess.run([sys.executable,str(ROOT/'scripts/configure_rig.py'),'--pitch-deg',str(pitch),'--roll-deg',str(roll),'--config',str(folder)],check=True,stdout=subprocess.DEVNULL)
                robot=json.loads((folder/'robot.json').read_text())
                mount=Rotation.from_quat(robot['imu_in_body_quaternion_xyzw'])
                model=ET.parse(folder/'model.sdf')
                board=np.array(list(map(float,model.findtext(".//link[@name='vio_rig']/pose").split())))
                np.testing.assert_allclose(Rotation.from_euler('xyz',board[3:]).as_matrix(),mount.as_matrix(),atol=1e-12)
                tof=np.array(list(map(float,model.findtext(".//sensor[@name='down_tof']/pose").split())))
                np.testing.assert_allclose((mount*Rotation.from_euler('xyz',tof[3:])).apply([1.,0.,0.]),[0,0,-1],atol=1e-12)
                np.testing.assert_allclose(board[:3]+mount.apply(tof[:3]),robot['tof_in_body_m'],atol=1e-12)
                # A single physical body point reprojects on the same native
                # image row, including when the physical baseline is vertical.
                point=mount.inv().apply(np.array([2.,.1,-.5])-board[:3])
                right=T.copy();right[1,3]-=baseline
                left_cam=T[:3,:3].T@(point-T[:3,3]);right_cam=right[:3,:3].T@(point-right[:3,3])
                left_uv=left_cam[:2]/left_cam[2]*K[:2]+K[2:]
                right_uv=right_cam[:2]/right_cam[2]*K[:2]+K[2:]
                self.assertAlmostEqual(left_uv[1],right_uv[1],places=10)
                self.assertAlmostEqual(left_uv[0]-right_uv[0],K[0]*baseline/left_cam[2],places=10)
                body=Rotation.from_euler('z',.6)
                optical=mount*Rotation.from_matrix(T[:3,:3])
                self.assertTrue(bearing_visible(.6,.6,body,optical,K,(1280,800)))
                self.assertFalse(bearing_visible(.6,.6+np.pi,body,optical,K,(1280,800)))
                if abs(roll)==90:
                    image_horizontal=mount.apply(T[:3,0])
                    self.assertGreater(abs(image_horizontal[2]),.99)


if __name__=='__main__':unittest.main()
