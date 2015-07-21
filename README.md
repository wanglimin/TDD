# Trajectory-Pooled Deep-Convolutional Descriptors (TDD)
Here we provide the code for the extraction of Trajectory-Pooled Deep-Convolutional Descriptors (TDD), from the following paper:

    Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors
    Limin Wang, Yu Qiao, and Xiaou Tang, in CVPR, 2015

#### Two-stream CNN models trained on the UCF101 dataset
First, we provide our trained two-stream CNN models on the split1 of UCF101 dataset, which achieve the recognition accuracy of 84.7%

["Spatial net model"](http://mmlab.siat.ac.cn/tdd/spatial.caffemodel) </br> 
["Spatial net prototxt"](http://mmlab.siat.ac.cn/tdd/spatial_cls.prototxt) </br>
["Temporal net model"](http://mmlab.siat.ac.cn/tdd/temporal.caffemodel) </br>
["Temporal net prototxt"](http://mmlab.siat.ac.cn/tdd/temporal_cls.prototxt)

#### TDD demo code
Here, a matlab demo code for TDD extraction is released.

- Step 1: Improved Trajectory Extraction </br>
You need download our modified iDT feature code and compile it by yourself. [Improved Trajectories](https://github.com/wanglimin/improved_trajectory)
- Step 2: TVL1 Optical Flow Extraction </br>
You need download our dense flow code and compile it by yourself. [Dense Flow](https://github.com/wanglimin/dense_flow)
- Step 3: Mat Caffe  </br>
You need download the public caffe toolbox. [Caffe](https://github.com/BVLC/caffe)
- Step 4: TDD Extraction </br>
Now you can run the matlab file "script_demo.m" to extract TDD features.


#### Questions
Contact 
- [Limin Wang](http://wanglimin.github.io/)

