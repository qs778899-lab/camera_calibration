import cv2
import numpy as np
import json
import os
import csv
from datetime import datetime
from threading import Thread, Lock
import time
from queue import Queue
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3, SO3
from my_utils.robot_utils import robot_fk
from record_episode import RealEnv
from my_utils.aruco_util import get_marker_pose,set_aruco_dict
from my_kalmen_filter import KalmenFilter




def same_position(pose1, pose2, angle_threshold=0.1, translation_threshold=0.01):
    """
    Compare two poses to determine if they are effectively the same position.
    
    Args:
        pose1, pose2: Marker poses to compare
        angle_threshold: Maximum angle difference in radians (default: 0.1)
        translation_threshold: Maximum translation difference in meters (default: 0.01)
    
    Returns:
        bool: True if poses are considered same, False otherwise
    """
    if pose1 is None or pose2 is None:
        return False
        
    R_diff = np.dot(pose1.R.T, pose2.R)
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    translation_norm = np.linalg.norm(pose1.t - pose2.t)
    return angle < angle_threshold and translation_norm < translation_threshold

class PositionMap:
    def __init__(self, id_list,camera_num):
        """
        Initialize position tracking for multiple markers.
        
        Args:
            id_list: List of marker IDs to track
        """
        self.position_map = {}
        self.overall_map = []
        
        self.camera_num = camera_num
        self.filter_list = {str(id): KalmenFilter() for id in id_list}
        self.temp_map = {str(id): [] for id in id_list}

        

    def reset_temp_map(self,marker_id):
        self.temp_map[marker_id] = []

    def filter_pose(self, marker_id, marker_pose):
        """Apply Kalman filter to marker pose."""
        marker_id = str(marker_id)
        if marker_id not in self.filter_list:
            self.filter_list[marker_id] = KalmenFilter()
            return marker_pose
            
        self.filter_list[marker_id].new_markerpose(marker_pose)
        self.filter_list[marker_id].Kalman_Filter()
        return self.filter_list[marker_id].get_pose()

    def update_position(self, marker_id, marker_pose):
        """Update marker position if significantly different from previous position."""
        marker_id = str(marker_id)
        
        if marker_pose is None:
            # 如果marker_pose为None，则将该marker_id的位置设置为None
            self.temp_map[marker_id].append(None)
            return

        filtered_pose = self.filter_pose(marker_id, marker_pose)
        
        if (marker_id in self.position_map and 
            self.position_map[marker_id] is not None and
            same_position(self.position_map[marker_id], filtered_pose)):
            # 如果marker_pose与之前的位置相同，则不更新位置
            self.temp_map[marker_id].append(self.position_map[marker_id])
            return

        self.temp_map[marker_id].append(filtered_pose)
        
    def combine_temp_map(self,marker_id):
        if len(self.temp_map[marker_id]) != self.camera_num:
            return
        # 去除None
        self.temp_map[marker_id] = [pose for pose in self.temp_map[marker_id] if pose is not None]
        if len(self.temp_map[marker_id]) == 0:
            # 如果temp_map中没有数据，则将position_map中的数据设置为None
            self.position_map[marker_id] = None
        else:
            self.position_map[marker_id] = self.temp_map[marker_id][0]

        self.reset_temp_map(marker_id)

    def get_position(self, marker_id):
        marker_id  = str(marker_id)
        return self.position_map[marker_id]
    
    def add2overall_map(self):
        self.overall_map.append(self.position_map.copy())





class MarkerTracker:
    def __init__(self, camera_names=["camera1", "camera3"],marker_json_path="marker_info.json"):
        '''
        camera1: eye-in-hand camera
        camera3: fixed camera        
        '''
        with open(marker_json_path, 'r') as f: #读取含有所有markers的信息文件
            self.marker_info = json.load(f)
        self.env = RealEnv(robot_names=['robot1'], camera_names=camera_names)
        
        self.id_list = list(self.marker_info.keys()) #可能有多个markers可以追踪
        self.position_map = PositionMap(self.id_list, len(camera_names))

        self.corner = {str(camera_name): {str(id): None for id in self.id_list} for camera_name in camera_names}

        print("begin calibrating cameras...")
        self.res_se3 = auto_regist_camera(0) #将marker_id=0传入，只追踪这个
        
        # Thread communication
        self.running = True
        self.lock = Lock()
        self.position_queue = Queue()
        self._thread = None
        
        self.image_queues = {cam: None for cam in camera_names}
        self._show_images = False

    def start_tracking(self, show_images=False):
        """Start tracking in separate thread"""
        self._show_images = show_images
        if self._thread is None:
            self._thread = Thread(target=self._process_data)
            self._thread.daemon = True
            self._thread.start()

    def get_image(self,camera_name):
        return self.image_queues[camera_name]

    def _process_data(self):
        """Process data in separate thread"""
        while self.running:
            obs = self.env.get_observation()
            
            for cam_name, img in obs['images'].items():
                img_copy = img.copy()
                
                # 处理标记检测
                for marker_id in self.id_list:
                    marker_pose, corners = get_marker_pose(
                        img_copy, 
                        marker_size=self.marker_info[marker_id]["marker_size"],
                        id=marker_id, 
                        aruco_dict=set_aruco_dict(self.marker_info[marker_id]["aruco_dict"]),
                        intrinsics_name="intrinsics_1" if cam_name == "camera1" else "intrinsics_3"
                    )
                    # convert to robot base frame
                    if cam_name == "camera3" and marker_pose is not None: #
                        marker_pose = self.res_se3 * marker_pose

                    if cam_name == "camera1" and marker_pose is not None:#
                        marker_pose = robot_fk(self.env.robots['robot1'], marker_pose)

                    with self.lock:
                        self.position_map.update_position(marker_id, marker_pose)
                        self.position_map.combine_temp_map(marker_id)
                        if corners is not None and marker_id in self.id_list:
                            self.corner[cam_name][marker_id] = corners

                with self.lock:
                    self.image_queues[cam_name] = img_copy
                   
            with self.lock:
                self.position_map.add2overall_map()

    def show_image(self, camera_name,marker_id,goal_corner=None):
        """
        显示指定相机的图像，返回按键值
        在主线程中调用此方法
        """
        if not self._show_images:
            return None
        img = None
        with self.lock:
            img = self.get_image(camera_name)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # draw corner line，在检测到marker时才可能会运行
            if goal_corner is not None and marker_id in self.corner[camera_name]:
                for (x1, y1), (x2, y2) in zip(self.corner[camera_name][marker_id], goal_corner):
                   cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)
            cv2.imshow(camera_name, img)
            return cv2.waitKey(1)
        return None

    def stop(self):
        """Stop tracking"""
        self.running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._show_images:
            cv2.destroyAllWindows()

    def get_marker_position(self, marker_id):
        """Get marker position thread-safely"""
        with self.lock:
            if marker_id is None:
                return None
            return self.position_map.get_position(marker_id)

    def get_all_positions(self):
        """Get all positions thread-safely"""
        with self.lock:
            return self.position_map.position_map.copy()

    def get_overall_map(self):
        """Get position history thread-safely"""
        with self.lock:
            return self.position_map.overall_map.copy()


#! 已知camera1(eye-in-hand)的手眼标定矩阵: 从末端执行器ee到相机cam , T_ee_cam 
transformations = SE3([
    [0.99994846, -0.00489269, -0.00889571, -0.03275156],
    [0.00641473,  0.98363436,  0.18006193, -0.06321087],
    [0.00786914, -0.18010971,  0.98361505,  0.07478308],
    [0.0,         0.0,         0.0,          1.0       ]
    ]) # Eye-hand calibration matrix

def auto_regist_camera(marker_id_input):
    '''
    Calibrate transformation between camera1 (eye-in-hand) and camera3 (fixed).
    
    Args:
        marker_id_input: ID of ArUco marker visible to both cameras
    
    Returns:
        SE3: Transformation from robot base to camera3(fixed)
    '''
    marker_id_input = str(marker_id_input)

    def setup_camera_env(camera_name):
        """Create environment and position map for a single camera."""
        env = RealEnv(robot_names=['robot1'], camera_names=[camera_name], enable_robot=False)  #? 创建camera时不要重复创建robot
        position_map = PositionMap([marker_id_input], 1)
        return env, position_map

    def get_marker_detection(env, camera_name, marker_info):
        """Get marker pose from camera image."""
        obs = env.get_observation()
        img = obs['images'].get(camera_name)
        if img is None:
            return None, None
        img = img.copy()
        return get_marker_pose(
            img,
            marker_size=marker_info[marker_id_input]["marker_size"],
            id=marker_id_input,
            aruco_dict=set_aruco_dict(marker_info[marker_id_input]["aruco_dict"]),
            intrinsics_name="intrinsics_1" if camera_name == "camera1" else "intrinsics_3"
        )

    # Setup environments and load marker info
    with open('marker_info.json', 'r') as f:
        marker_info = json.load(f)
    
    env1, position_map1 = setup_camera_env("camera1")
    env2, position_map2 = setup_camera_env("camera3")

    os.makedirs("record_tests", exist_ok=True)
    csv_filename = datetime.now().strftime("%Y%m%d_%H%M%S_camera_pose.csv")
    csv_path = os.path.join("record_tests", csv_filename)

    def pose_to_str(pose):
        if pose is None:
            return ""
        translation = pose.t
        rpy = pose.rpy(order='xyz')
        return (
            f"[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}, "
            f"{rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f}]"
        )

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "iteration",
            "camera1_pose",
            "camera3_pose",
        ])
        camera_order = ["camera1", "camera3"]

        # Collect and filter marker poses
        iteration = 0
        # while True:
        for iteration in range(1000):
            # iteration += 1
            cam_results = {}
            for env, camera_name, position_map in [
                (env1, "camera1", position_map1),
                (env2, "camera3", position_map2)
            ]:
                marker_pose, _ = get_marker_detection(env, camera_name, marker_info) #这里面会调用get qpos
                position_map.update_position(marker_id_input, marker_pose) #用卡尔曼滤波更新实际marker的position
                position_map.combine_temp_map(marker_id_input)   #它确保每一轮都在收齐所有相机的观测之后，才更新“当前标记姿态”。这样既能兼顾多相机的数据，又避免用单个相机的残缺结果覆盖掉其他信息。
                cam_results[camera_name] = marker_pose

            writer.writerow([
                iteration,
                pose_to_str(cam_results.get("camera1")),
                pose_to_str(cam_results.get("camera3")),
            ])

        # Calculate transformation
        camera1_pose = position_map1.position_map[marker_id_input]
        camera3_pose = position_map2.position_map[marker_id_input]
        
        # camera1_pose: T_cam1_marker
        # camera3_pose.inv(): T_marker_cam3 

        print("camera1_pose: ", camera1_pose)
        print("camera3_pose: ", camera3_pose)

        res = camera1_pose * camera3_pose.inv()  # Camera-to-camera transformation
        #res: T_cam1_cam3
        robot_ref = env2.robots.get('robot1') or RealEnv._shared_robots.get('robot1')
        if robot_ref is None:
            raise RuntimeError("Unable to obtain shared robot instance for camera calibration.")
        
        robot_pose = robot_ref.get_pose_se3() #函数定义在record_episode.py中, 里面会调用dobot.get_pose()
        # print("robot_pose: ", robot_pose)   

        # if hasattr(robot_ref, "dobot"):
        #     raw_pose = robot_ref.dobot.get_pose()
        #     # print(
        #     #     "robot pose raw (x,y,z,rx,ry,rz): "
        #     #     f"{raw_pose[0]:.4f}, {raw_pose[1]:.4f}, {raw_pose[2]:.4f}, "
        #     #     f"{raw_pose[3]:.4f}, {raw_pose[4]:.4f}, {raw_pose[5]:.4f}"
        #     # )
        #     rotation_list = [raw_pose[3],raw_pose[4], raw_pose[5]]
        #     R_xyz = R.from_euler('xyz', rotation_list, degrees=True)
        #     raw_pose_matrix = SE3.Rt(
        #         SO3(R_xyz.as_matrix()),
        #         [raw_pose[0]/1000, raw_pose[1]/1000, raw_pose[2]/1000], 
        #         check=False
        #     )
        #     raw_pose_matrix_array = np.array(raw_pose_matrix)
            # print("raw_pose_matrix_array", raw_pose_matrix_array)

        # robot_pose: T_base_ee
        # transformations: T_ee_cam1
        # res: T_cam1_cam3
        res = robot_pose * transformations * res
        # 新res: T_base_cam3

        translation_res = res.t
        rpy_res = res.rpy(order='xyz')
        print(
            "T_base_camera3 (matrix):\n", res,
            "\nT_base_camera3 (x,y,z,rx,ry,rz): "
            f"x={translation_res[0]:.4f}, y={translation_res[1]:.4f}, z={translation_res[2]:.4f}, "
            f"rx={rpy_res[0]:.4f}, ry={rpy_res[1]:.4f}, rz={rpy_res[2]:.4f}"
        )

        

        return res





def main():
    tracker = MarkerTracker()
    tracker.start_tracking(show_images=True)
    time.sleep(2)
    
    try:
        while True:
            # 显示所有相机图像
            key1 = tracker.show_image("camera1", marker_id=0)  #追踪marker_id=0的图像，显示camera1的追踪结果
            key3 = tracker.show_image("camera3", marker_id=0)  #追踪marker_id=0的图像，显示camera3的追踪结果
            
            # 检查退出条件
            if key1 == ord('q'):
                break
            
            # 获取位置数据
            # positions = tracker.get_all_positions()
            # print("\rCurrent positions:", positions, end='')
            
    except KeyboardInterrupt:
        print("\nStopping tracker...")
    finally:
        tracker.stop()




if __name__ == "__main__":
    main()

