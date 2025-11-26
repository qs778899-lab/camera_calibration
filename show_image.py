import rospy
import time
import numpy as np
from spatialmath import SE3
import json
import sys
import os
import argparse


from marker_tracker import same_position
from aruco_data.id_info import IDInfoList, IDInfo
from marker_tracker import MarkerTracker
from my_utils.aruco_util import get_marker_pose,set_aruco_dict
from my_utils.robot_utils import robot_move,robot_fk,robot_ee2marker
from my_utils.myRobotSaver import MyRobotSaver,read_movement,replay_movement
from follow_aruco import *
from marker_tracker import *
from record_episode import init_robot  # 导入统一的 init_robot 函数




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dino Bot Control Script')
    parser.add_argument('--camera1', type=str, default="camera1", help='Name of first camera')
    parser.add_argument('--camera2', type=str, default="camera3", help='Name of second camera')
    parser.add_argument('--robot', type=str, default="robot1", help='Robot name')
    args = parser.parse_args()

    # ROS 节点已在 record_episode 导入时初始化，不需要重复初始化
    
    # 初始化机器人和夹爪
    robot = init_robot(args.robot)

    # 尝试初始化 Dobot gripper
    gripper = None
    # try:
    #     sys.path.append('/home/erlin/work/labgrasp')
    #     from dobot_gripper import DobotGripper
    #     gripper = DobotGripper(robot.dobot)
    #     gripper.connect(init=True)
    # except Exception as e:
    #     print(f"Warning: Could not initialize gripper: {e}")
    #     gripper = None
    
    tracker = MarkerTracker(camera_names=[args.camera1, args.camera2])
    tracker.start_tracking(show_images=True)
    time.sleep(3)
    Action = MarkerAction(robot, gripper, tracker)
    while True:
        key1 = tracker.show_image(args.camera1, Action.maker_id, Action.goal_corner)
        key2 = tracker.show_image(args.camera2, Action.maker_id, Action.goal_corner)

        if key1 == ord('c') or key2 == ord('c'):
            Action.set_goal_object()