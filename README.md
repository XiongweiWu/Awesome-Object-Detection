# Awesome-Object-Detection

A list of awesome object detection resources.

Recently we released *[survey](https://arxiv.org/abs/1908.03673)* to the community. In this survey, we systematically analyze the existing object detection frameworks and organize the survey into three major parts: (i) detection components, (ii) learning strategies, and (iii) applications & benchmarks. In the survey, we cover a variety of factors affecting the detection performance in detail, such as detector architectures, feature learning, proposal generation, sampling strategies, etc. Finally, we discuss several future directions to facilitate and spur future research for visual object detection with deep learning.  

After completing this survey, we decided to release the collected resource of object detection. We will keep updating our survey as well as this resource collection, since this area moves too fast. If you have any questions or suggestions, please feel free to contact us.

**Table of Contents**

* [1. Generic Object Detection](#1-Generic-Object-Detection)
    - [1.1 Two-stage Detection Algorithms](#11-Two-stage-Detection)
    - [1.2 One-stage Detection Algorithms](#11-One-stage-Detection)
* [2. Face Detection](#2-Supervised-SR)
* [3. Pedestrian Detection](#3-Unsupervised-SR)
* [4. MSCOCO Datasets](#4-SR-Datasets)
* [5. Pascal VOC Datasets](#5-SR-Metrics)
* [6. Other Resources](#6-Other-Resources)

**Citing this work**

If this repository is useful, please cite our [survey](https://arxiv.org/abs/1908.03673).

```
@article{wu2019recent,
    title={Recent Advances in Deep Learning for Object Detection},
    author={Xiongwei Wu, Doyen Sahoo, Steven C.H. Hoi},
    journal={arXiv preprint arXiv:1908.03673},
    year={2019}
}
```



## 1. Generic Object Detection

### 1.1 Two-stage Detection

**2012 ECCV**

1. **Anchored Neighborhood Regression for Fast Example-Based Super-Resolution**, *Timofte, Radu; De, Vincent; Van Gool, Luc*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2013/html/Timofte_Anchored_Neighborhood_Regression_2013_ICCV_paper.html)], [[Project](http://www.vision.ee.ethz.ch/~timofter/ICCV2013_ID1774_SUPPLEMENTARY/index.html)], `ANR`, `GR`



**2012 CVPR**


**2012 NeurIPS**


**2013 ICCV**


**2013 CVPR**


**2013 NeurIPS**


**2014 ECCV**

1. **Spatial pyramid pooling in deep convolutional networks for visual recognition**, *K. He, X. Zhang, S. Ren, J. Sun*, [[Arxiv](https://arxiv.org/pdf/1406.4729)], [[Caffe-Matlab](https://github.com/ShaoqingRen/SPP_net)], `SPP-Net`


**2014 CVPR**

1. **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation**, *R. Girshick, J. Donahue, T. Darrell, J. Malik*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)],[[Supplementary](http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf)], [[Caffe](https://github.com/rbgirshick/rcnn)], `RCNN`

**2014 NeurIPS**


**2015 ICCV**

1. **Fast r-cnn**, *R. Girshick*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)], [[Caffe-Python](https://github.com/rbgirshick/fast-rcnn)], `Fast R-CNN`



**2015 CVPR**


**2015 NeurIPS**

1. **Faster r-cnn: Towards real-time object detection with region proposal networks**, *S. Ren, K. He, R. Girshick, J. Sun*, [[OpenAccess](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)],[[Arxiv](https://arxiv.org/pdf/1506.01497)],[[Caffe-Matlab](https://github.com/shaoqingren/faster_rcnn)], [[Caffe-Python](https://github.com/rbgirshick/py-faster-rcnn)],[[Pytorch](https://github.com/jwyang/faster-rcnn.pytorch)], [[TensorFlow](https://github.com/endernewton/tf-faster-rcnn)], [[MXNet](https://github.com/apache/incubator-mxnet/tree/master/example/rcnn)], Faster R-CNN`


**2016 ECCV**


**2016 CVPR**

1. **Hypernet: Towards accurate region proposal generation and joint object detection**, *T. Kong, A. Yao, Y. Chen, F. Sun*, [[OpenAccess](https://zpascal.net/cvpr2016/Kong_HyperNet_Towards_Accurate_CVPR_2016_paper.pdf)], `HyperNet`
2. **Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks**, *S. Bell, C. Lawrence Zitnick, K. Bala, R. Girshick*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/papers/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.pdf)], `ION`
3. **Object detection from video tubelets with convolutional neural networks**, *K. Kang, W. Ouyang, H. Li, X. Wang*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kang_Object_Detection_From_CVPR_2016_paper.pdf)], [[Caffe](https://github.com/myfavouritekk/T-CNN)], `T-CNN`
4. **Instance-aware semantic segmentation via multitask network cascades**, *J. Dai, K. He, J. Sun*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)], [[Caffe](https://github.com/daijifeng001/MNC)], `MNC`
5. **Adaptive object detection using adjacency and zoom prediction**, *Y. Lu, T. Javidi, S. Lazebnik*, [[Arxiv](https://arxiv.org/abs/1512.07711)], [[Caffe](https://github.com/luyongxi/az-net)], `AZ-Net`



**2016 NeurIPS**

1. **R-fcn: Object detection via region-based fully convolutional networks**, *J. Dai, Y. Li, K. He, J. Sun*, [[OpenAccess](https://papers.nips.cc/paper/6465-r-fcn-object-detection-via-region-based-fully-convolutional-networks.pdf)], [[Caffe-Matlab](https://github.com/daijifeng001/R-FCN)], [[Caffe-Python](https://github.com/YuwenXiong/py-R-FCN)], `R-FCN`

**2016 Arxiv**

1. **Beyond skip connections: Top-down modulation for object detection**, *A. Shrivastava, R. Sukthankar, J. Malik, A. Gupta*, [[Arxiv](https://arxiv.org/abs/1612.06851)], `TDM`

**2017 ICCV**

1. **Mask R-CNN**, *K. He, G. Gkioxari, P. Dollar, R. Girshick*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)],[[Caffe2](https://github.com/facebookresearch/Detectron)], [[Slides](http://kaiminghe.com/iccv17maskrcnn/maskrcnn_iccv2017_oral_kaiminghe.pdf)], `Mask R-CNN`
2. **Denet: Scalable real-time object detection with directed sparse sampling**, *L. Tychsen-Smith, L. Petersson*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tychsen-Smith_DeNet_Scalable_Real-Time_ICCV_2017_paper.pdf)],[[Theano](https://github.com/lachlants/denet)], `DeNet`
3. **Deformable convolutional networks**, *J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, Y. Wei*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)],[[MXNet](https://github.com/msracver/Deformable-ConvNets)], `DCN`
 
 

**2017 CVPR**

1. **Feature pyramid networks for object detection**, *T.Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, S. Belongie*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)], [[Caffe2](https://github.com/facebookresearch/Detectron)], `FPN`




**2017 NeurIPS**

**2018 ECCV**


**2018 CVPR**

1. **Cascade r-cnn: Delving into high quality object detection**, *Z. Cai, N. Vasconcelos*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/zhaoweicai/cascade-rcnn)], [[Caffe2](https://github.com/zhaoweicai/Detectron-Cascade-RCNN)] `Cascade R-CNN`
2. **Detnet: A backbone network for object detection**, *Z. Li, C. Peng, G. Yu, X. Zhang, Y. Deng, J. Sun*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zeming_Li_DetNet_Design_Backbone_ECCV_2018_paper.pdf)], [[Pytorch*](https://github.com/guoruoqian/DetNet_pytorch)], `DetNet`
3. **An analysis of scale invariance in object detection–snip**, *B. Singh, L. S. Davis*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Singh_An_Analysis_of_CVPR_2018_paper.pdf)], [[MXNet](https://github.com/mahyarnajibi/SNIPER)], `SNIP`
4. **Multi-scale location-aware kernel representation for object detection**, *H. Wang, Q. Wang, M. Gao, P. Li, W. Zuo*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Scale_Location-Aware_Kernel_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/Hwang64/MLKP)], `MLKR`


**2018 NeurIPS**

1. **Metaanchor: Learning to detect objects with customized anchors**, *T. Yang, X. Zhang, Z. Li, W. Zhang, J. Sun*,  [[OpenAccess](https://papers.nips.cc/paper/7315-metaanchor-learning-to-detect-objects-with-customized-anchors)], `MetaAnchor`


**2019 ICCV**


**2019 CVPR**
1. **Mask scoring r-cnn**, *Z. Huang, L. Huang, Y. Gong, C. Huang, X. Wang*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)], [[Pytorch](https://github.com/zjhuang22/maskscoring_rcnn)], `Mask Scoring R-CNN`



**2019 NeurIPS**


**2019 AAAI**
1. **Derpn: Taking a further step toward more general object detection**, *L. J. Z. X. Lele Xie, Yuliang Liu*, [[OpenAccess](https://www.aaai.org/ojs/index.php/AAAI/article/view/4936/4809)], [[Caffe](https://github.com/HCIILAB/DeRPN)], `DeRPN`














### 1.2 One-stage Detection

**Before 2014**
1. **Overfeat: Integrated recognition, localization and detection using convolutional networks**, *P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, Y. LeCun*, in: arXiv preprint arXiv:1312.6229, 2013.  [[Arxiv](https://arxiv.org/pdf/1312.6229)], [[Torch](https://github.com/sermanet/OverFeat)], `Overfeat`


**2014 ECCV**


**2014 CVPR**


**2014 NeurIPS**


**2015 ICCV**


**2015 CVPR**


**2015 NeurIPS**

**2016 ECCV**
1. **SSD: Single shot multibox detector**, *W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, A. C. Berg*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Object_Detection_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/weiliu89/caffe/tree/ssd)], `SSD` 


**2016 CVPR**
 1. **You only look once: Unified, real-time object detection**, *J. Redmon, S. Divvala, R. Girshick, A. Farhadi*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)], [[DarkNet](https://github.com/pjreddie/darknet)], `YOLO`




**2016 NeurIPS**


**2017 ICCV**
1. **Focal loss for dense object detection**, *T.Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)], [[Caffe2](https://github.com/facebookresearch/Detectron)], `RetinaNet` 
2. **Dsod: Learning deeply supervised object detectors from scratch**, *Z. Shen, Z. Liu, J. Li, Y.-G. Jiang, Y. Chen, X. Xue*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Shen_DSOD_Learning_Deeply_ICCV_2017_paper.pdf)], [[Caffe](https://github.com/szq0214/DSOD)], `DSOD`

**2017 CVPR**

1. **Yolo9000: better, faster, stronger**, *J. Redmon, A. Farhadi*, [[OpenAccess](http://zpascal.net/cvpr2017/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)], [[DarkNet](https://pjreddie.com/darknet/yolo)], `YOLOv2`


**2017 NeurIPS**

**2018 ECCV**

1. **Cornernet: Detecting objects as paired keypoints**, *H. Law, J. Deng*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)], [[Pytorch](https://github.com/princeton-vl/CornerNet)], `CornerNet`
2. **Receptive field block net for accurate and fast object detection**, *S. Liu, D. Huang, Y. Wang*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Songtao_Liu_Receptive_Field_Block_ECCV_2018_paper.html)], [[Pytorch](https://github.com/ruinmessi/RFBNet)], `RFBNet`
3. **Deep feature pyramid reconfiguration for object detection**, *T. Kong, F. Sun, W. Huang, H. Liu*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Kong_Deep_Feature_Pyramid_ECCV_2018_paper.pdf)], [[Pytorch](https://github.com/ruinmessi/RFBNet)], `RFBNet`


 

**2018 CVPR**

1. **Single-shot refinement neural network for object detection**, *S. Zhang, L. Wen, X. Bian, Z. Lei, S. Z. Li*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)], [[Caffe](https://github.com/sfzhang15/RefineDet)], `RefineDet`
2. **Scale-transferrable object detection**, *P. Zhou, B. Ni, C. Geng, J. Hu, Y. Xu*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1376.pdf)], [[Pytorch](https://github.com/arvention/STDN)], `STDN`

**2018 NeurIPS**



**2019 ICCV**


**2019 CVPR**

1. **Feature selective anchor-free module for single-shot object detection**, *C. Zhu, Y. He, M. Savvides*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.pdf)], `FSFA`

**2019 NeurIPS**

**2019 Arxiv**

1. **Objects as points**, *X. Zhou, D. Wang, P. Krahenb ¨ uhl*, [[Arxiv](https://arxiv.org/pdf/1904.07850)], [[Pytorch](https://github.com/xingyizhou/CenterNet)], `CenterNet`
2. **Centernet: Keypoint triplets for object detection**, *K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang, Q. Tian*, [[Arxiv](https://arxiv.org/pdf/1904.08189)], [[Pytorch](https://github.com/Duankaiwen/CenterNet)], `CenterNet`


## 2. Face Detection

**2015 Arxiv**

**2017 CVPR**

1. **Finding tiny faces**, *P. Hu, D. Ramanan*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_Finding_Tiny_Faces_CVPR_2017_paper.pdf)], [[MatConvNet](https://github.com/peiyunh/tiny)], `S3FD`

**2017 ICCV**

1. **S3fd: Single shot scale-invariant face detector**, *S. Zhang, X. Zhu, Z. Lei, H. Shi, X. Wang, S. Z. Li*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)], [[Caffe](https://github.com/sfzhang15/SFD)], `S3FD`
2. **Recurrent scale approximation for object detection in cnn**, *Y. Liu, H. Li, J. Yan, F. Wei, X. Wang, X. Tang*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Recurrent_Scale_Approximation_ICCV_2017_paper.pdf)], [[Caffe](https://github.com/sciencefans/RSA-for-object-detection)], `RSA`


**2018 CVPR**

1. **Seeing small faces from robust anchors perspective**, *C. Zhu, R. Tao, K. Luu, M. Savvides*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3468.pdf)]


## 3. Pedestrian Detection
 
**2015 CVPR**
1. **Taking a deeper look at pedestrians**, *J. Hosang, M. Omran, R. Benenson, B. Schiele*,  [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hosang_Taking_a_Deeper_2015_CVPR_paper.pdf)]


**2016 ECCV**

1. **A unified multi-scale deep convolutional neural network for fast object detection**, *Z. Cai, Q. Fan, R. S. Feris, N. Vasconcelos*, [[OpenAccess](http://www.eccv2016.org/files/posters/P-2B-38.pdf)], [[Caffe](https://github.com/zhaoweicai/mscnn)], `MSCNN`

**2016 CVPR**

1. **Exploit all the layers: Fast and accurate cnn object detector with scale dependent pooling and cascaded rejection classifiers**, *F. Yang, W. Choi, Y. Lin*, [ [OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Yang_Exploit_All_the_CVPR_2016_paper.html)], `SDP-CRC`


**2017 CVPR**

1. **Accurate single stage detector using recurrent rolling convolution**, *J. Ren, X. Chen, J. Liu, W. Sun, J. Pang, Q. Yan, Y.-W. Tai, L. Xu*, [[OpenAccess](http://zpascal.net/cvpr2017/Ren_Accurate_Single_Stage_CVPR_2017_paper.pdf)], [[Caffe](https://github.com/xiaohaoChen/rrc_detection)], `RRC`

 **2018 TMM**

1. **Scale-aware fast r-cnn for pedestrian detection**, *J. Li, X. Liang, S. Shen, T. Xu, J. Feng, S. Yan*,  [[Arxiv](https://arxiv.org/pdf/1510.08160)], `SAF R-CNN`


















