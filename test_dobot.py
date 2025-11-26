import sys
import rospy
sys.path.append('/home/erlin/work/labgrasp')
from simple_api import SimpleApi
rospy.init_node('dobot_controller', anonymous=True)  #
dob = SimpleApi("192.168.5.1", 29999)
dob.clear_error()
resp = dob.enable_robot()
print("EnableRobot response:", resp)
print(dob.get_pose())

dob2 = SimpleApi("192.168.5.2", 29999)
dob2.clear_error()
resp = dob2.enable_robot()
print("EnableRobot response:", resp)
print(dob2.get_pose())