# Trajectory-Pooled Deep-Convolutional Descriptors
Here we provide the code for the extraction of Trajectory-Pooled Deep-Convolutional Descriptors (TDD), from the following paper:

    Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors
    Limin Wang, Yu Qiao, and Xiaou Tang, in CVPR, 2015
### Updates
- Dec 24, 2015
  * Release the second version of TDD (branch: cudnn2.0) compatible with latest [caffe toolbox](https://github.com/yjxiong/caffe). Due to speedup brought by cudnn2.0 or above, TDD extraction is becoming more efficient.
- Jul 21, 2015
  * Release the first version TDD (branch: master) compatible with an older version of [caffe toolbox](https://github.com/wanglimin/caffe).

### Two-stream CNN models trained on the UCF101 dataset
First, we provide our trained two-stream CNN models on the split1 of UCF101 dataset, which achieve the recognition accuracy of 84.7%

["Spatial net model (v1)"](http://mmlab.siat.ac.cn/tdd/spatial.caffemodel) </br> 
["Spatial net prototxt (v1)"](http://mmlab.siat.ac.cn/tdd/spatial_cls.prototxt) </br>
["Temporal net model (v1)"](http://mmlab.siat.ac.cn/tdd/temporal.caffemodel) </br>
["Temporal net prototxt (v1)"](http://mmlab.siat.ac.cn/tdd/temporal_cls.prototxt)

### TDD demo code
Here, a matlab demo code for TDD extraction is provided.

- **Step 1**: Improved Trajectory Extraction </br>
You need download our modified iDT feature code and compile it by yourself. [Improved Trajectories](https://github.com/wanglimin/improved_trajectory)
- **Step 2**: TVL1 Optical Flow Extraction </br>
You need download our dense flow code and compile it by yourself. [Dense Flow](https://github.com/wanglimin/dense_flow)
- **Step 3**: Matcaffe  </br>
You need download the public caffe toolbox. Our TDD code is compatatible with the latest version of [parallel caffe toolbox](https://github.com/yjxiong/caffe). </br>
**Note that you need to download the models in the new proto format:** </br>
["Spatial net model (v2)"](http://mmlab.siat.ac.cn/tdd/spatial_v2.caffemodel) ["Temporal net model (v2)"](http://mmlab.siat.ac.cn/tdd/temporal_v2.caffemodel) </br>
- **Step 4**: TDD Extraction </br>
Now you can run the matlab file "script_demo.m" to extract TDD features.


### Questions
Contact 
- [Limin Wang](http://wanglimin.github.io/)

