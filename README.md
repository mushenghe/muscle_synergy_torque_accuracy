# muscle_synergy_torque_accuracy
project to explore the relationship between muscle activation patterns and torque-perceptual accuracy
# HealerBaxter

This project aims to use deep learning and manipulation platform to train a two-armed robot to play the piano based on the human emotion.

<img src="images/emotion_detection.gif" height=300px width="420" align="left"/>
<img src="images/piano_play.gif" height=300px  width="420" align="left"/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>

This repository provides Tenserflow code for traning and testing the emotion detection policies with deep learning in real time, as well as the ROS python code for manipulate the robot to play the piano.

For an explanation of the neural network model structure please see my [portfolio post](https://mushenghe.github.io/)

## Requirements
* Ubuntu 18.04
* ROS melodic
* [Kaggle Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
## Hardware:
* [Baxter Robot](https://robots.ieee.org/robots/baxter/)
* [Soft roll-up piano](https://www.amazon.com/iMeshbean-Portable-Flexible-Electronic-Keyboard/dp/B07W929C1Y/ref=sr_1_14?keywords=roll+up+piano+61+keys&qid=1577941987&s=toys-and-games&sr=1-14)
* [AprilTags](http://wiki.ros.org/apriltag_ros)
* [AA Batteries](https://www.amazon.com/AmazonBasics-Performance-Alkaline-Batteries-Count/dp/B00MNV8E0C?ref_=s9_apbd_orecs_hd_bw_bQMcmB&pf_rd_r=5RQFAAJHZ8EJ758E09VB&pf_rd_p=a50b8358-02f3-5eb1-a8ca-3ec2c519285c&pf_rd_s=merchandised-search-10&pf_rd_t=BROWSE&pf_rd_i=389577011)

## Getting Started
### Installation
This implementation requires the following dependencies:
* install moveit
```
sudo apt install ros-melodic-moveit
```
* install apriltag_ros package
```
sudo apt install ros-melodic-apriltag-ros
```
* install viewer package for ROS image topics
```
sudo apt install ros-melodic-image-view
```
* install image rectification and color processing package for ROS
```
sudo apt install ros-melodic-image-proc
```
* install tensorflow and keras
    + Check if your Python environment is already configured: 
    ```
        sudo apt update
        sudo apt install python3-dev python3-pip
        sudo pip3 install -U virtualenv  # system-wide install
    ```

    + Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it: 
    ```
        virtualenv --system-site-packages -p python3 ./venv
    ```
    
    +  Activate the virtual environment using a shell-specific command:
    ```
        source ./venv/bin/activate  # sh, bash, ksh, or zsh
    ```
    
    + Install packages within a virtual environment without affecting the host system setup. Start by upgrading pip: 
    ```
        pip install --upgrade pip
        pip list  # show packages installed within the virtual environment
    ```
    
    + Install the TensorFlow pip package:
    ```
        pip install --upgrade tensorflow
    ```
    
    + Verify the install:
    ```
        python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```
    
    + install Keras
    ```
        pip install keras
    ```

### Quickstart
+ Create a new workspace, download healer.rosinstall file and use this to download the code
    - Some code comes from forked repositories, with fixes to make it compile on Ubuntu 18.04 with melodic
+ Builld the workspace
    - Use catkin_make -DCMAKE_BUILD_TYPE = Release to enable optimizations and improve the resulting performance
+ Connect and enable the robot, open the left camera
+ Launch the piano playing launch file, manipulate the robot to the initial configuration, ready for emotion detection and piano playing
+ Use pre-trained model webcam to detect the emotion, when sad face is detected, baxter start play the piano

Instructions:
+ Create and build workspace
```
mkdir -p HealerBaxter/src
cd HealerBaxter/src
// download healer.rosinstall
wstool init
wstool merge healer.rosinstall
wstool update
cd ..
catkin_make 
source devel/setup.bash
```

+ Connect Robot and manipulate it to the initial configuration
```
source src/healer/config/connect.sh
roslaunch healer piano_playing.launch
```
+ Start detecting emotion, use python I/O file to send the detected emotion to baxter and start playing the piano
```
source ~/venv/bin/activate
cd src/healer/emotion_detection/
python test_model.py 
rosrun healer piano_play 
```
## Future work
+ Train model to recognize and distinguish people, make the system only work for one master
+ Make baxter to have flexibility to play any series of notes based on the user input
+ Make docker container to bring up all the dependencies