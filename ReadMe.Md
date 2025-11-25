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

        查看usb连接的 RealSense 相机
        rs-enumerate-devices

        启动camera1
        roslaunch realsense2_camera rs_camera.launch \
            camera:=camera1 \
            serial_no:=231522072272 \
            enable_color:=true \
            enable_depth:=true

        启动camera3
        roslaunch realsense2_camera rs_camera.launch \
            camera:=camera3 \
            serial_no:=250122078799 \
            enable_color:=true \
            enable_depth:=true


        查看camera1的节点是否成功发布，比如查看彩色图像的发布频率
        rostopic hz /camera1/color/image_raw

        查看camera3的节点是否成功发布，比如查看彩色图像的发布频率
        rostopic hz /camera3/color/image_raw

        note：
        不需要roscore, roslaunch 自动启动

2. 启动

        python show_image.py --camera1 camera1 --camera2 camera3
