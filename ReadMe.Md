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

2. transformations： T_ee_cam, camera1(eye-in-hand)的手眼标定矩阵。 NOTE: 这个参数在多个文件中有设置。

3. 相近内参： 在camera_intrinsics.py中有设置。

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

2. 测试
        
        测试dobot连接是否正正常:
        /bin/python3  test_dobot.py


3. 运行主程序进行标定计算

        python show_image.py --camera1 camera1 --camera2 camera3 (暂时不用)

        /bin/python3   show_image.py --camera1 camera1 --camera2 camera3  (暂时不用)

        /bin/python3  marker_tracker.py 


        验证⚠️: 多次只移动marker位置，如果输出矩阵数值不变则标定正确。 或者移动camera1的位置。


4. 标定结果记录

                标定一组:

                T_base_camera3 (matrix):
                0.1024    0.624    -0.7747    1.225     
                0.9944   -0.08573   0.06237  -0.2917    
                -0.0275   -0.7767   -0.6292    0.4094    
                0         0         0         1         
                T_base_camera3 (x,y,z,rx,ry,rz): x=1.2249, y=-0.2917, z=0.4094, rx=-1.4082, ry=-0.8863, rz=-3.0428


                T_base_camera3 (matrix):
                0.1008    0.6519   -0.7516    1.211     
                0.9948   -0.07593   0.06752  -0.2965    
                -0.01305  -0.7545   -0.6561    0.4325    
                0         0         0         1         
                T_base_camera3 (x,y,z,rx,ry,rz): x=1.2110, y=-0.2965, z=0.4325, rx=-1.4174, ry=-0.8505, rz=-3.0390


                T_base_camera3 (matrix):
                0.09935   0.6601   -0.7446    1.2       
                0.9943   -0.09438   0.049    -0.2809    
                -0.03792  -0.7452   -0.6657    0.4313    
                0         0         0         1         
                T_base_camera3 (x,y,z,rx,ry,rz): x=1.2000, y=-0.2809, z=0.4313, rx=-1.4214, ry=-0.8399, rz=-3.0681


                标定二组: 

                T_base_camera3 (matrix):
                0.1454    0.7146   -0.6842    1.084     
                0.9894   -0.1096    0.09575  -0.2729    
                -0.006537 -0.6909   -0.723     0.4143    
                0         0         0         1     
                T_base_camera3 (x,y,z,rx,ry,rz): x=1.0842, y=-0.2729, z=0.4143, rx=-1.3701, ry=-0.7535, rz=-3.0099

                T_base_camera3 (matrix):
                0.1436    0.6988   -0.7007    1.097     
                0.9896   -0.09552   0.1075   -0.2839    
                0.008209 -0.7089   -0.7053    0.3931    
                0         0         0         1         
                T_base_camera3 (x,y,z,rx,ry,rz): x=1.0974, y=-0.2839, z=0.3931, rx=-1.3682, ry=-0.7764, rz=-2.9903

                T_base_camera3 (matrix):
                0.1389    0.717    -0.683     1.078     
                0.9899   -0.08189   0.1153   -0.2906    
                0.02676  -0.6922   -0.7212    0.3976    
                0         0         0         1         
                T_base_camera3 (x,y,z,rx,ry,rz): x=1.0783, y=-0.2906, z=0.3976, rx=-1.3795, ry=-0.7519, rz=-2.9830









5. 获取相机内参


        印出所有连接设备的内参详细信息
        rs-enumerate-devices -c

        获取相机实际使用的分辨率
        rostopic echo /camera3/color/image_raw --noarr -n1 | grep -E "height|width"
        比如：h 720 * w 1280
        rostopic echo /camera1/color/image_raw --noarr -n1 | grep -E "height|width"

        找到对应分辨率的内参参数
        注意是 Intrinsic of "Color"前缀的参数
        cam3_k: (h 720 * w 1280)
        [907.5684,   0,       637.8553],
        [  0,      907.5656, 360.8097],
        [  0,        0,         1     ]

        cam1_k: (h 720 * w 1280)
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

2. 识别不到marker, AttributeError: 'NoneType' object has no attribute 'inv':

        原因1：
        marker摆放位置不好导致相机里的marker不够清晰

        原因2:
        重新走一遍完整的初始化流程


3. 标定结果数值不准

        原因1:
        marker不够平整，纸面没有完全紧贴平面

        原因2:
        内参数值没有对应
