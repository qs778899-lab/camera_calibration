import rospy
import time
import numpy as np
import sys
import collections
import dm_env
import cv2
from spatialmath import SE3
from spatialmath import SO3

from my_utils.record_utils import Recorder, ImageRecorder

# Ensure ROS node exists before importing Dobot utilities
try:
    rospy.init_node('dobot_controller', anonymous=True)
except rospy.exceptions.ROSException:
    pass

sys.path.append('/home/erlin/work/labgrasp')
from simple_api import SimpleApi
from dobot_gripper import DobotGripper


class DummyRobot:
    """Fallback robot used when the real robot is unavailable."""
    def __init__(self):
        print("WARNING: Using dummy robot (no real robot connected)")
        self.dobot = None

    def get_pose_se3(self):
        print("fake pose in DummyRobot")
        return SE3()

    def get_joint_positions(self):
        print("fake joint positions in DummyRobot")
        return [0.0] * 6


class DobotRobotWrapper:
    """Wrapper class providing the interface expected by the rest of the code."""
    def __init__(self, robot_ip="192.168.5.1", robot_port=29999):
        self.dobot = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.dobot = SimpleApi(robot_ip, robot_port)
                self.dobot.clear_error()
                self.dobot.enable_robot()
                self.dobot.stop()
                # 启动力传感器
                self.dobot.enable_ft_sensor(1)
                time.sleep(1)
                # 力传感器置零
                self.dobot.six_force_home()
                time.sleep(1)
                print("dobot 连接成功")
                break
            except Exception as e:
                print(f"连接失败，第 {attempt + 1} 次重试: {e}")
                if self.dobot is not None:
                    try:
                        self.dobot.close()
                    except Exception:
                        pass
                    self.dobot = None
                time.sleep(2)
        else:
            raise RuntimeError("无法连接 Dobot，已重试多次失败。")

    def get_pose_se3(self):
        """Get current end-effector pose as SE3 object"""
        pose = self.dobot.get_pose()
        if pose is None:
            return SE3()
        x, y, z, rx, ry, rz = pose[:6]
        translation = np.array([x / 1000.0, y / 1000.0, z / 1000.0])
        rotation = SO3.RPY(np.deg2rad([rx, ry, rz]))
        return SE3.Rt(rotation.R, translation)

    def get_joint_positions(self):
        return self.dobot.get_joint_positions()


_ROBOT_CACHE = {}


def init_robot(name, use_dummy_if_fail=True):
    """Initialize robot by name - compatible with original interface"""
    robot_configs = {
        'robot1': {'ip': '192.168.5.1', 'port': 29999},
    }
    config = robot_configs.get(name, robot_configs['robot1'])

    if name in _ROBOT_CACHE and _ROBOT_CACHE[name] is not None: #避免重复初始化dobot
        return _ROBOT_CACHE[name]

    if use_dummy_if_fail:
        try:
            robot = DobotRobotWrapper(config['ip'], config['port'])
            _ROBOT_CACHE[name] = robot
            return robot
        except Exception as e:
            print(f"WARNING: Failed to connect to robot {name} at {config['ip']}:{config['port']}")
            print(f"Error: {e}")
            print("Using dummy robot instead. Camera functions will work, but robot functions will not.")
            dummy = DummyRobot()
            _ROBOT_CACHE[name] = dummy
            return dummy
    else:
        robot = DobotRobotWrapper(config['ip'], config['port'])
        _ROBOT_CACHE[name] = robot
        return robot


class RealEnv:
    """
    Observation space: {"qpos": Concat[ arm (6),          # absolute joint position

                        "images": {"camera1": (480x640x3),        # h, w, c, dtype='uint8'
                                   "camera2": (480x640x3),         # h, w, c, dtype='uint8'
                                   ....}
    """

    def __init__(self, robot_names, camera_names=['camera1'], enable_robot=True):
        try:
            rospy.init_node('ik_step', anonymous=True)
        except:
            pass
        self.robot_names = robot_names
        self.robots = {}
        self.robot_infos = {}
        self._robots_enabled = enable_robot
        if enable_robot:
            for i, name in enumerate(robot_names):
                self.robots[name] = init_robot(name)
                self.robot_infos[name] = Recorder(name, init_node=False)
        else:
            for name in robot_names:
                self.robots[name] = None
                self.robot_infos[name] = None
        self.image_recorder = ImageRecorder(init_node=False, camera_names=camera_names)

        time.sleep(2)

    def get_qpos(self, robot_names): #? 逻辑
        qpos = []
        for robot_name in robot_names:
            recorder = self.robot_infos.get(robot_name)
            if recorder is None or recorder.qpos is None:
                continue
            qpos.append(recorder.qpos)

        if not qpos:
            print("qpos is empty!")
            return np.array([])
        return np.concatenate(qpos)

    def get_data(self, robot_names): #? 逻辑
        data = []
        for robot_name in robot_names:
            recorder = self.robot_infos.get(robot_name)
            if recorder is None or recorder.data is None:
                continue
            data.append(recorder.data)
        return data

    def get_images(self):
        return self.image_recorder.get_images()

    def get_observation(self): #? 逻辑
        obs = collections.OrderedDict()
        qpos = self.get_qpos(self.robot_names)
        obs['qpos'] = qpos if qpos.size != 0 else None
        obs['images'] = self.get_images()
        return obs


if __name__ == "__main__":
    robot_names = ['robot1']
    camera_name = ['camera1', 'camera3']
    env = RealEnv(robot_names, camera_name)
    while True:
        obs = env.get_observation()
        print(obs['qpos'])

        # Display images from all cameras
        for cam_name, img in obs['images'].items():
            cv2.imshow(cam_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        time.sleep(1)
