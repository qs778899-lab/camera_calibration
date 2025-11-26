# 相机标定: 使用已标定好的camera1 (eye-in-hand) 来标定camera3 (fixed)


## 环境配置:

    sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
    sudo apt-get install ros-noetic-realsense2-camera

    /bin/python3 -m pip install dm-env opencv-contrib-python spatialmath-python rospkg catkin_pkg numpy h5py ipython message_filters Cython pyrealsense2
    
    sudo mkdir -p /etc/apt/keyrings  # 注册公钥
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list    # 添加源（Ubuntu 20.04 用 focal，22.04 用 jammy）
    sudo apt update
    sudo apt install librealsense2-utils




## 关键参数设置(根据实际情况调整):

1. data.json: 需要识别二维码的信息

2. transformations： T_ee_cam, camera1(eye-in-hand)的手眼标定矩阵


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

2. 测试dobot连接是否正正常
        
        /bin/python3  test_dobot.py


3. 运行主程序进行标定计算

        python show_image.py --camera1 camera1 --camera2 camera3 (暂时不用)

        /bin/python3   show_image.py --camera1 camera1 --camera2 camera3  (暂时不用)

        /bin/python3  marker_tracker.py 


        验证⚠️: 多次只移动marker位置，如果输出矩阵数值不变则标定正确。 或者移动camera1的位置？


4. 标定结果记录

        例子1：
        -0.08477   0.6161   -0.7831    0.7618    
        0.9132   -0.2663   -0.3084   -0.4537    
        -0.3985   -0.7413   -0.54      0.4132    
        0         0         0         1  

5. 获取相机内参


        印出所有连接设备的内参详细信息
        rs-enumerate-devices -c

        获取相机实际使用的分辨率
        rostopic echo /camera3/color/image_raw --noarr -n1 | grep -E "height|width"
        比如：h 720 * w 1280
        rostopic echo /camera1/color/image_raw --noarr -n1 | grep -E "height|width"

        找到对应分辨率的内参参数
        注意是 Intrinsic of "Color"前缀的参数
        cam3_k:
        [907.5684,   0,       637.8553],
        [  0,      907.5656, 360.8097],
        [  0,        0,         1     ]

        cam1_k:
        [906.8782,   0,       636.7186],
        [  0,      907.1544, 368.1572],
        [  0,        0,         1     ]






## 示例效果

![Demo Result](./demo_results/demo1.jpg)



## 常见问题:

1. 相机设备无法正常初始化：

        原因1：
        sudo fuser -v /dev/video*
        kill +进程号

2. 
