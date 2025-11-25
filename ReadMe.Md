# 相机标定: 使用已标定好的camera1 (eye-in-hand) 来标定camera3 (fixed)


## 环境配置:

sudo apt-get install ros-$ROS_DISTRO-realsense2-camera

sudo apt-get install ros-noetic-realsense2-camera



## 关键参数设置:

1. data.json: 需要识别二维码的信息


## 标定操作流程:

0. 放置 ArUco 标记 
如(ID=0, 尺寸0.1m)，确保两个相机都能看到标记

1. 连接相机

    rs-enumerate-devices: 查看usb连接的 RealSense 相机


    roslaunch realsense2_camera rs_camera.launch \
        camera:=camera1 \
        serial_no:=231522072272 \
        enable_color:=true \
        enable_depth:=true


    rostopic hz /camera1/color/image_ra: 查看camera1的节点是否成功发布，比如查看彩色图像的发布频率