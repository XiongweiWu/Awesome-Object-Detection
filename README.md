# Awesome-Object-Detection

*Last Update: 2019/08/16*

**TODO:**

- [ ] Add SOTA Table
- [ ] Fix the link and format issues

A list of awesome object detection resources.

Recently we released *[survey](https://arxiv.org/abs/1908.03673)* to the community. In this survey, we systematically analyze the existing object detection frameworks and organize the survey into three major parts: (i) detection components, (ii) learning strategies, and (iii) applications & benchmarks. In the survey, we cover a variety of factors affecting the detection performance in detail, such as detector architectures, feature learning, proposal generation, sampling strategies, etc. Finally, we discuss several future directions to facilitate and spur future research for visual object detection with deep learning.  

After completing this survey, we decided to release the collected resource of object detection. We will keep updating our survey as well as this resource collection, since this area moves too fast. If you have any questions or suggestions, please feel free to contact us.

**Table of Contents**

* [1. Generic Object Detection](#1-Generic-Object-Detection)
    - [1.1 Two-stage Detection Algorithms](#11-Two-stage-Detection)
    - [1.2 One-stage Detection Algorithms](#12-One-stage-Detection)
* [2. Face Detection](#2-Face-Detection)
* [3. Pedestrian Detection](#3-Pedestrian-Detection)
* [4. Benchmarks](#4-Benchmarks)
    - [4.1 Generic Detection Datasets](#41-Generic-Detection-Datasets)
    - [4.2 Face Detection Datasets](#42-Face-Detection-Datasets)
    - [4.3 Pedestrian Detection Datasets](#41-Pedestrian-Detection-Datasets)
* [5. SOTA](#5-SOTA)
    - [5.1 Pascal VOC](#51-Pascal-VOC)
    - [5.2 MSCOCO](#52-MSCOCO)
* [6. Future Work](#6-Future-Work)
    - [6.1 Anchor Design](#61-Anchor-Design)
        - [6.1.1 Anchor-Free Methods](#611-Anchor-Free-Methods)
        - [6.1.2 Anchor-Refinement Methods](#611-Anchor-Refinement-Methods)
    - [6.2 AutoML Detection](#62-AutoML-Detection)
    - [6.3 Low-shot Detection](#63-Low-shot-Detection)
    - [6.4 Others](#64-Others)
* [7. Other Resources](#6-Other-Resources)

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

**2014 CVPR**

1. **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation**, *R. Girshick, J. Donahue, T. Darrell, J. Malik*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)],[[Supplementary](http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf)], [[Caffe](https://github.com/rbgirshick/rcnn)], `RCNN`

**2014 ECCV**

1. **Spatial pyramid pooling in deep convolutional networks for visual recognition**, *K. He, X. Zhang, S. Ren, J. Sun*, [[Arxiv](https://arxiv.org/pdf/1406.4729)], [[Caffe-Matlab](https://github.com/ShaoqingRen/SPP_net)], `SPP-Net`

**2015 CVPR**

1. **Deepid-net: Deformable deep convolutional neural networks for object detection**,*W. Ouyang, X. Wang, X. Zeng, S. Qiu, P. Luo, Y. Tian, H. Li, S. Yang, Z. Wang, C.-C. Loy*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ouyang_DeepID-Net_Deformable_Deep_2015_CVPR_paper.pdf)]
2. **segdeepm: Exploiting segmentation and context in deep neural networks for object detection**, *Y. Zhu, R. Urtasun, R. Salakhutdinov, S. Fidler*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2015/ext/3B_028_ext.pdf)]
3.  **Deformable part models are convolutional neural networks**, *R. Girshick, F. Iandola, T. Darrell, J. Malik*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Girshick_Deformable_Part_Models_2015_CVPR_paper.pdf)]

**2015 ICCV**

1. **Fast r-cnn**, *R. Girshick*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)], [[Caffe-Python](https://github.com/rbgirshick/fast-rcnn)], `Fast R-CNN`
2. **Object detection via a multi-region and semantic segmentation-aware cnn model**, *S. Gidaris, N. Komodakis*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Gidaris_Object_Detection_via_ICCV_2015_paper.pdf)], [[Caffe](https://github.com/gidariss/mrcnn-object-detection)], `MR-CNN`
3. **Deepproposal: Hunting objects by cascading deep convolutional layers**, *A. Ghodrati, A. Diba, M. Pedersoli, T. Tuytelaars, L. Van Gool*, [[OpenAccess](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ghodrati_DeepProposal_Hunting_Objects_ICCV_2015_paper.pdf)], [[MatConvnet](https://github.com/aghodrati/deepproposal)], `Deepproposal`

**2015 NeurIPS**

1. **Faster r-cnn: Towards real-time object detection with region proposal networks**, *S. Ren, K. He, R. Girshick, J. Sun*, [[OpenAccess](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)],[[Arxiv](https://arxiv.org/pdf/1506.01497)],[[Caffe-Matlab](https://github.com/shaoqingren/faster_rcnn)], [[Caffe-Python](https://github.com/rbgirshick/py-faster-rcnn)],[[Pytorch](https://github.com/jwyang/faster-rcnn.pytorch)], [[TensorFlow](https://github.com/endernewton/tf-faster-rcnn)], [[MXNet](https://github.com/apache/incubator-mxnet/tree/master/example/rcnn)], `Faster R-CNN`


**2016 CVPR**

1. **Hypernet: Towards accurate region proposal generation and joint object detection**, *T. Kong, A. Yao, Y. Chen, F. Sun*, [[OpenAccess](https://zpascal.net/cvpr2016/Kong_HyperNet_Towards_Accurate_CVPR_2016_paper.pdf)], `HyperNet`
2. **Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks**, *S. Bell, C. Lawrence Zitnick, K. Bala, R. Girshick*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/papers/Bell_Inside-Outside_Net_Detecting_CVPR_2016_paper.pdf)], `ION`
3. **Object detection from video tubelets with convolutional neural networks**, *K. Kang, W. Ouyang, H. Li, X. Wang*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kang_Object_Detection_From_CVPR_2016_paper.pdf)], [[Caffe](https://github.com/myfavouritekk/T-CNN)], `T-CNN`
4. **Instance-aware semantic segmentation via multitask network cascades**, *J. Dai, K. He, J. Sun*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)], [[Caffe](https://github.com/daijifeng001/MNC)], `MNC`
5. **Adaptive object detection using adjacency and zoom prediction**, *Y. Lu, T. Javidi, S. Lazebnik*, [[Arxiv](https://arxiv.org/abs/1512.07711)], [[Caffe](https://github.com/luyongxi/az-net)], `AZ-Net`
6. **Training region-based object detectors with online hard example mining**, *A. Shrivastava, A. Gupta, R. Girshick*,  [[OpenAccess](https://zpascal.net/cvpr2016/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)], [[Caffe](https://github.com/abhi2610/ohem)], `OHEM` 
7. **Locnet: Improving localization accuracy for object detection**, *S. Gidaris, N. Komodakis*, [[OpenAccess](https://zpascal.net/cvpr2016/Gidaris_LocNet_Improving_Localization_CVPR_2016_paper.pdf)], [[Matlab](https://github.com/gidariss/LocNet)], `LocNet` 
8. **Craft objects from images**, *B. Yang, J. Yan, Z. Lei, S. Z. Li*, [[OpenAccess](https://yan-junjie.github.io/publication/dblp-confcvpr-yang-yll-16/dblp-confcvpr-yang-yll-16.pdf)], [[Caffe](https://github.com/byangderek/CRAFT)], `CRAFT`

**2016 ECCV**

1. **Contextual priming and feedback for faster r-cnn**, *A. Shrivastava, A. Gupta*, [[OpenAccess](http://abhinavsh.info/papers/pdfs/context_priming_feedback.pdf)]
2. **Gated bi-directional cnn for object detection**, *X. Zeng, W. Ouyang, B. Yang, J. Yan, X. Wang*, [[OpenAccess](http://www.cs.toronto.edu/~byang/papers/gbd_eccv16.pdf)]


**2016 NeurIPS**

1. **R-fcn: Object detection via region-based fully convolutional networks**, *J. Dai, Y. Li, K. He, J. Sun*, [[OpenAccess](https://papers.nips.cc/paper/6465-r-fcn-object-detection-via-region-based-fully-convolutional-networks.pdf)], [[Caffe-Matlab](https://github.com/daijifeng001/R-FCN)], [[Caffe-Python](https://github.com/YuwenXiong/py-R-FCN)], `R-FCN`

**2016 Others**

1. **Beyond skip connections: Top-down modulation for object detection**, *A. Shrivastava, R. Sukthankar, J. Malik, A. Gupta*, in: arXiv preprint arXiv:1612.06851, 2016. [[Arxiv](https://arxiv.org/abs/1612.06851)], `TDM`
2. **A multipath network for object detection**, *S. Zagoruyko, A. Lerer, T.-Y. Lin, P. O. Pinheiro, S. Gross, S. Chintala, P. Dollar*, in: BMVC, 2016. [[Arxiv](https://arxiv.org/abs/1604.02135)], [[Torch](https://github.com/facebookresearch/multipathnet)], `MultiPathNet`
3. **Pvanet: deep but lightweight neural networks for real-time object detection**, *K.-H. Kim, S. Hong, B. Roh, Y. Cheon, M. Park*, in: arXiv preprint arXiv:1608.08021, 2016. [[Arxiv](https://arxiv.org/abs/1608.08021)], [[Caffe](https://github.com/sanghoon/pva-faster-rcnn)], `PVANet`

**2017 CVPR**

1. **Feature pyramid networks for object detection**, *T.Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, S. Belongie*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)], [[Caffe2](https://github.com/facebookresearch/Detectron)], `FPN`
2. **Perceptual generative adversarial networks for small object detection**, *J. Li, X. Liang, Y. Wei, T. Xu, J. Feng, S. Yan*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.pdf)], `PGAN`
3. **A-fast-rcnn: Hard positive generation via adversary for object detection**,  *X. Wang, A. Shrivastava, A. Gupta*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)], [Caffe](https://github.com/xiaolonw/adversarial-frcnn)],`A-Fast-RCNN`
4. **Mimicking very efficient network for object detection**, *Q. Li, S. Jin, J. Yan*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf)]
5. **Learning non-maximum suppression**, *J. Hosang, R. Benenson, B. Schiele*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.pdf)], [[TensorFlow](https://github.com/hosang/gossipnet)]
6. **Speed/accuracy trade-offs for modern convolutional object detectors**, *J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara, A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, et al.*, [[OpenAccess](http://zpascal.net/cvpr2017/Huang_SpeedAccuracy_Trade-Offs_for_CVPR_2017_paper.pdf)], [[TensorFlow](https://github.com/tensorflow/models/tree/master/research/object_detection)]


**2017 ICCV**

1. **Mask R-CNN**, *K. He, G. Gkioxari, P. Dollar, R. Girshick*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)],[[Caffe2](https://github.com/facebookresearch/Detectron)], [[Slides](http://kaiminghe.com/iccv17maskrcnn/maskrcnn_iccv2017_oral_kaiminghe.pdf)], `Mask R-CNN`
2. **Denet: Scalable real-time object detection with directed sparse sampling**, *L. Tychsen-Smith, L. Petersson*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tychsen-Smith_DeNet_Scalable_Real-Time_ICCV_2017_paper.pdf)],[[Theano](https://github.com/lachlants/denet)], `DeNet`
3. **Deformable convolutional networks**, *J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, Y. Wei*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf)],[[MXNet](https://github.com/msracver/Deformable-ConvNets)], `DCN`
4. **Couplenet: Coupling global structure with local parts for object detection**, *Y. Zhu, C. Zhao, J. Wang, X. Zhao, Y. Wu, H. Lu*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_CoupleNet_Coupling_Global_ICCV_2017_paper.pdf)],[[Caffe](https://github.com/tshizys/CoupleNet)], `CoupleNet`
5. **Spatial memory for context reasoning in object detection**, *X. Chen, A. Gupta*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Spatial_Memory_for_ICCV_2017_paper.pdf)], `SMN`
6. **Soft-nms – improving object detection with one line of code**, *N. Bodla, B. Singh, R. Chellappa, L. S. Davis*, [[OpenAccess](http://www.cs.umd.edu/~bharat/snms.pdf)], [[Caffe](https://github.com/bharatsingh430/soft-nms)]

**2017 Others**

1. **Light-head rcnn: In defense of two-stage object detector**, *Z. Li, C. Peng, G. Yu, X. Zhang, Y. Deng, J. Sun*, in: arXiv preprint arXiv:1711.07264, 2017. [[Arxiv](https://arxiv.org/abs/1711.07264)], [[Pytorch](https://github.com/Sundrops/pytorch-faster-rcnn)], [[TensorFlow](https://github.com/zengarden/light_head_rcnn)]
2. **Zoom out-and-in network with recursive training for object proposal**, *H. Li, Y. Liu, W. Ouyang, X. Wang*, in: arXiv preprint arXiv:1702.05711, 2017. [[Arxiv](https://arxiv.org/abs/1702.05711)]


**2018 CVPR**

1. **Cascade r-cnn: Delving into high quality object detection**, *Z. Cai, N. Vasconcelos*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/zhaoweicai/cascade-rcnn)], [[Caffe2](https://github.com/zhaoweicai/Detectron-Cascade-RCNN)] `Cascade R-CNN`
2. **Detnet: A backbone network for object detection**, *Z. Li, C. Peng, G. Yu, X. Zhang, Y. Deng, J. Sun*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zeming_Li_DetNet_Design_Backbone_ECCV_2018_paper.pdf)], [[Pytorch*](https://github.com/guoruoqian/DetNet_pytorch)], `DetNet`
3. **An analysis of scale invariance in object detection–snip**, *B. Singh, L. S. Davis*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Singh_An_Analysis_of_CVPR_2018_paper.pdf)], [[MXNet](https://github.com/mahyarnajibi/SNIPER)], `SNIP`
4. **Multi-scale location-aware kernel representation for object detection**, *H. Wang, Q. Wang, M. Gao, P. Li, W. Zuo*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Scale_Location-Aware_Kernel_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/Hwang64/MLKP)], `MLKR`
5. **Feature selective networks for object detection**, *Y. Zhai, J. Fu, Y. Lu, H. Li*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhai_Feature_Selective_Networks_CVPR_2018_paper.pdf)]
6. **Pseudo mask augmented object detection**, *X. Zhao, S. Liang, Y. Wei*, [[OpenAccess](https://www.zpascal.net/cvpr2018/Zhao_Pseudo_Mask_Augmented_CVPR_2018_paper.pdf)]
7. **Structure inference net: Object detection using scene-level context and instance-level relationships**, *Y. Liu, R. Wang, S. Shan, X. Chen*, [[OpenAccess](https://arxiv.org/abs/1807.00119)], [[TensorFlow](https://github.com/choasup/SIN)], `SIN`
8. **Relation networks for object detection**, *H. Hu, J. Gu, Z. Zhang, J. Dai, Y. Wei*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Relation_Networks_for_CVPR_2018_paper.pdf)], [[MXNet](https://github.com/msracver/Relation-Networks-for-Object-Detection)]
9. **Path Aggregation Network for Instance Segmentation**, *S. Liu, L. Qi, H. Qin, J. Shi and J. Jia*, [[OpenAccess](http://jiaya.me/papers/panet_cvpr18.pdf)], [[Pytorch](https://github.com/ShuLiu1993/PANet)]

**2018 ECCV**

1. **Acquisition of localization confidence for accurate object detection**, *B. Jiang, R. Luo, J. Mao, T. Xiao, Y. Jiang*, [[OpenAccess](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.pdf)], [[Pytorch](https://github.com/vacancy/PreciseRoIPooling)], `IoU-Net`
2. **Revisiting rcnn: On awakening the classification power of faster rcnn**, *B. Cheng, Y. Wei, H. Shi, R. Feris, J. Xiong, T. Huang*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bowen_Cheng_Revisiting_RCNN_On_ECCV_2018_paper.pdf)], [[MXNet](https://github.com/bowenc0221/Decoupled-Classification-Refinement)]
3. **Learning region features for object detection**, *J. Gu, H. Hu, L. Wang, Y. Wei, J. Dai*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jiayuan_Gu_Learning_Region_Features_ECCV_2018_paper.pdf)]
4. **Deep regionlets for object detection**, *H. Xu, X. Lv, X. Wang, Z. Ren, R. Chellappa*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hongyu_Xu_Deep_Regionlets_for_ECCV_2018_paper.pdf)]
5. **Context refinement for object detection**, *Z. Chen, S. Huang, D. Tao*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhe_Chen_Context_Refinement_for_ECCV_2018_paper.pdf)]


**2018 NeurIPS**

1. **Metaanchor: Learning to detect objects with customized anchors**, *T. Yang, X. Zhang, Z. Li, W. Zhang, J. Sun*,  [[OpenAccess](https://papers.nips.cc/paper/7315-metaanchor-learning-to-detect-objects-with-customized-anchors)], `MetaAnchor`
2. **Sniper: Efficient multi-scale training**, *B. Singh, M. Najibi, L. S. Davis*, [[OpenAccess](https://papers.nips.cc/paper/8143-sniper-efficient-multi-scale-training.pdf)], [[MXNet](https://github.com/mahyarnajibi/SNIPER)], `SNIPER`

**2019 AAAI**

1. **Derpn: Taking a further step toward more general object detection**, *L. J. Z. X. Lele Xie, Yuliang Liu*, [[OpenAccess](https://www.aaai.org/ojs/index.php/AAAI/article/view/4936/4809)], [[Caffe](https://github.com/HCIILAB/DeRPN)], `DeRPN`
2. **Object Detection based on Region Decomposition and Assembly**, *S.-H Bae*, [[OpenAccess](https://arxiv.org/abs/1901.08225)], `R-DAD`

**2019 CVPR**

1. **Mask scoring r-cnn**, *Z. Huang, L. Huang, Y. Gong, C. Huang, X. Wang*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)], [[Pytorch](https://github.com/zjhuang22/maskscoring_rcnn)], `Mask Scoring R-CNN`
2. **Deformable convnets v2: More deformable, better results**, *S. L. Xizhou Zhu, Han Hu, J. Dai*,  [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.pdf)], [[MXNet](https://github.com/msracver/Deformable-ConvNets)], `DCNv2`
3. **Grid r-cnn**, *X. Lu, B. Li, Y. Yue, Q. Li, J. Yan*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Grid_R-CNN_CVPR_2019_paper.pdf)], [[mmdetection](https://github.com/open-mmlab/mmdetection)]
4. **Nas-fpn: Learning scalable feature pyramid architecture for object detection**, *G. Ghiasi, T.-Y. Lin, Q. V. Le*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghiasi_NAS-FPN_Learning_Scalable_Feature_Pyramid_Architecture_for_Object_Detection_CVPR_2019_paper.pdf)], [[TensorFlow](https://github.com/DetectionTeamUCAS/NAS_FPN_Tensorflow)], `NAS-FPN`
5. **Bounding Box Regression with Uncertainty for Accurate Object Detection**, *Y. He, C. Zhu, J. Wang, M. Savvides, X. Zhang*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.pdf)], [[Caffe2](https://github.com/yihui-he/KL-Loss)], `KL-Loss`
6. **Libra R-CNN: Towards Balanced Learning for Object Detection**, *J. Pang, K. Chen, J. Shi, H. Feng, W. Ouyang, D. Lin*, [[OpenAccess](http://dahua.me/publications/dhl19_librarcnn.pdf)], [[Pytorch](https://github.com/OceanPang/Libra_R-CNN)], [[mmdetection](https://github.com/open-mmlab/mmdetection)], `Libra R-CNN`
7. **Region Proposal by Guided Anchoring**, *J. Wang, K. Chen, S. Yang, C. C. Loy, D. Lin*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Region_Proposal_by_Guided_Anchoring_CVPR_2019_paper.pdf)], [[mmdetection](https://github.com/open-mmlab/mmdetection)]

**2019 ICCV**

1. **Rethinking imagenet pre-training**, *R. G. Kaiming He, P. Dollro*, [[OpenAccess](https://arxiv.org/abs/1811.08883)]

**2019 Others**

1. **Scale-aware trident networks for object detection**, *Y. Li, Y. Chen, N. Wang, Z. Zhang*, in: arXiv preprint arXiv:1901.01892, 2019. [[OpenAccess](https://arxiv.org/abs/1901.01892)], [[MXNet](https://github.com/TuSimple/simpledet)], `TridentNet`

**2019 NeurIPS**



### 1.2 One-stage Detection

**Before 2014**
1. **Overfeat: Integrated recognition, localization and detection using convolutional networks**, *P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, Y. LeCun*, in: arXiv preprint arXiv:1312.6229, 2013.  [[Arxiv](https://arxiv.org/pdf/1312.6229)], [[Torch](https://github.com/sermanet/OverFeat)], `Overfeat`

**2016 CVPR**
 1. **You only look once: Unified, real-time object detection**, *J. Redmon, S. Divvala, R. Girshick, A. Farhadi*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)], [[DarkNet](https://github.com/pjreddie/darknet)], `YOLO`

**2016 ECCV**
1. **SSD: Single shot multibox detector**, *W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, A. C. Berg*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Object_Detection_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/weiliu89/caffe/tree/ssd)], `SSD` 

**2017 CVPR**

1. **Yolo9000: better, faster, stronger**, *J. Redmon, A. Farhadi*, [[OpenAccess](http://zpascal.net/cvpr2017/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)], [[DarkNet](https://pjreddie.com/darknet/yolo)], `YOLOv2`
2. **Ron: Reverse connection with objectness prior networks for object detection**, *T. Kong, F. Sun, A. Yao, H. Liu, M. Lu, Y. Chen*, [[OpenAccess](http://zpascal.net/cvpr2017/Kong_RON_Reverse_Connection_CVPR_2017_paper.pdf)], [[Caffe](https://github.com/taokong/RON)], `RON`

**2017 ICCV**
1. **Focal loss for dense object detection**, *T.Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollar*, [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)], [[Caffe2](https://github.com/facebookresearch/Detectron)], `RetinaNet` 
2. **Dsod: Learning deeply supervised object detectors from scratch**, *Z. Shen, Z. Liu, J. Li, Y.-G. Jiang, Y. Chen, X. Xue*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Shen_DSOD_Learning_Deeply_ICCV_2017_paper.pdf)], [[Caffe](https://github.com/szq0214/DSOD)], `DSOD`

**2017 Others**

1. **Dssd: Deconvolutional single shot detector**, *C.-Y. Fu, W. Liu, A. Ranga, A. Tyagi, A. C. Berg*, in: arXiv preprint arXiv:1701.06659, 2017. [[OpenAccess](https://arxiv.org/abs/1701.06659)], [[Caffe](https://github.com/zchrissirhcz/caffe-dssd)], `DSSD`
2. **Residual features and unified prediction network for single stage detection**, *K. Lee, J. Choi, J. Jeong, N. Kwak*, in: arXiv preprint arXiv:1707.05031, 2017. [[OpenAccess](https://arxiv.org/abs/1707.05031)]
3. **Enhancement of ssd by concatenating feature maps for object detection**, *J. Jeong, H. Park, N. Kwak*, in: arXiv preprint arXiv:1705.09587, 2017. [[OpenAccess](https://arxiv.org/abs/1705.09587)]
4. **Fssd: Feature fusion single shot multibox detector**, *Z. Li, F. Zhou*, in: arXiv preprint arXiv:1705.1712.00960, 2017. [[OpenAccess](https://arxiv.org/abs/1712.00960)], `FSSD`
5. **Learning object detectors from scratch with gated recurrent feature pyramids**, *Z. Shen, H. Shi, R. Feris, L. Cao, S. Yan, D. Liu, X. Wang, X. Xue, T. S. Huang*, in: arXiv preprint arXiv:1712.00886, 2017. [[OpenAccess](https://arxiv.org/abs/1712.00886)], [[Caffe](https://github.com/szq0214/GFR-DSOD)]


**2018 CVPR**

1. **Single-shot refinement neural network for object detection**, *S. Zhang, L. Wen, X. Bian, Z. Lei, S. Z. Li*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)], [[Caffe](https://github.com/sfzhang15/RefineDet)], `RefineDet`
2. **Scale-transferrable object detection**, *P. Zhou, B. Ni, C. Geng, J. Hu, Y. Xu*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1376.pdf)], [[Pytorch](https://github.com/arvention/STDN)], `STDN`
3. **Single-shot object detection with enriched semantics**, *Z. Zhang, S. Qiao, C. Xie, W. Shen, B. Wang, A. L. Yuille*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Object_Detection_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/bairdzhang/des)], `DES`

**2018 ECCV**

1. **Cornernet: Detecting objects as paired keypoints**, *H. Law, J. Deng*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)], [[Pytorch](https://github.com/princeton-vl/CornerNet)], `CornerNet`
2. **Receptive field block net for accurate and fast object detection**, *S. Liu, D. Huang, Y. Wang*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Songtao_Liu_Receptive_Field_Block_ECCV_2018_paper.html)], [[Pytorch](https://github.com/ruinmessi/RFBNet)], `RFBNet`
3. **Deep feature pyramid reconfiguration for object detection**, *T. Kong, F. Sun, W. Huang, H. Liu*, [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Kong_Deep_Feature_Pyramid_ECCV_2018_paper.pdf)]

**2018 Others**

1. **YOLOv3: An Incremental Improvement**, *J. Redmon, A. Farhadi*, in: arXiv preprint arXiv:1804.02767, 2018. [[OpenAccess](https://pjreddie.com/media/files/papers/YOLOv3.pdf)], [[DarkNet](https://pjreddie.com/yolo/)], `YOLOv3`
2. **Mdssd: Multi-scale deconvolutional single shot detector for small objects**, *M. Xu, L. Cui, P. Lv, X. Jiang, J. Niu, B. Zhou, M. Wang*, in: arXiv preprint arXiv:1805.07009, 2018. [[Arxiv](https://arxiv.org/abs/1805.07009)], `MDSSD`


**2019 AAAI**

1. **M2det: A single-shot object detector based on multi-level feature pyramid network**, *Q. Zhao, T. Sheng, Y. Wang, Z. Tang, Y. Chen, L. Cai, H. Ling*, [[OpenAccess](https://qijiezhao.github.io/imgs/m2det.pdf)], [[Pytorch](https://github.com/qijiezhao/M2Det)], `M2Det`
2. **Gradient harmonized single-stage detector**, *Y. L. Buyu Li, X. Wang*, [[OpenAccess](https://aaai.org/ojs/index.php/AAAI/article/view/4877)], [[mmdetection ](https://github.com/libuyu/GHM_Detection)], `GHM`

**2019 CVPR**

1. **Feature selective anchor-free module for single-shot object detection**, *C. Zhu, Y. He, M. Savvides*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.pdf)], `FSFA`
2. **Scratchdet: Exploring to train single-shot object detectors from scratch**, *R. Zhu, S. Zhang, X. Wang, L. Wen, H. Shi, L. Bo, T. Mei*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_ScratchDet_Training_Single-Shot_Object_Detectors_From_Scratch_CVPR_2019_paper.pdf)], [[Caffe](https://github.com/KimSoybean/ScratchDet)], `Scratchdet`
3. **Bottom-up object detection by grouping extreme and center points**, *X. Zhou, J. Zhuo, P. Krahenbuhl*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_Bottom-Up_Object_Detection_by_Grouping_Extreme_and_Center_Points_CVPR_2019_paper.pdf)], [[Pytorch](https://github.com/xingyizhou/ExtremeNet)], `ExtremeNet`
4. **Towards Accurate One-Stage Object Detection with AP-Loss**,
*K. Chen, J. Li, W. Lin, J. See, J. Wang, L. Duan, Z. Chen, C. He, J. Zou*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Accurate_One-Stage_Object_Detection_With_AP-Loss_CVPR_2019_paper.pdf)], `AP-Loss`

**2019 ICCV**

1. **Fcos: Fully convolutional one-stage object detection**, *Z. Tian, C. Shen, H. Chen, T. He*, [[OpenAccess](https://arxiv.org/abs/1904.01355)], [[Pytorch](https://github.com/tianzhi0549/FCOS)], `FCOS`
2. **RepPoints: Point Set Representation for Object Detection**, *Z. Yang, S. Liu, H. Hu, L. Wang, S. Lin*, [[OpenAccess](https://arxiv.org/abs/1904.11490)], `RepPoints`

**2019 Others**

1. **Objects as points**, *X. Zhou, D. Wang, P. Krahenb ¨ uhl*, in: arXiv preprint arXiv:1904.07850, 2019, [[Arxiv](https://arxiv.org/pdf/1904.07850)], [[Pytorch](https://github.com/xingyizhou/CenterNet)], `CenterNet`
2. **Centernet: Keypoint triplets for object detection**, *K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang, Q. Tian*, in: arXiv preprint arXiv:1904.08189, 2019, [[Arxiv](https://arxiv.org/pdf/1904.08189)], [[Pytorch](https://github.com/Duankaiwen/CenterNet)], `CenterNet`
3. **CornerNet-Lite: Efficient Keypoint Based Object Detection**, *Hei Law, Yun Teng, Olga Russakovsky, Jia Deng*, in: arXiv preprint arXiv:1904.08900, 2019, [[OpenAccess](https://arxiv.org/abs/1904.08900)], [[Pytorch](https://github.com/princeton-vl/CornerNet-Lite)], `CornerNet-Lite`
4. **Revisiting Feature Alignment for One-stage Object Detection**, *Y. Chen, C. Han, N. Wang, Z. Zhang*, in: arXiv preprint arXiv:1908.01570, 2019, [[OpenAccess](https://arxiv.org/abs/1908.01570)], `AlignDet`
5. **PosNeg-Balanced Anchors with Aligned Features for Single-Shot Object Detection**, *Qiankun Tang, Shice Liu, Jie Li, Yu Hu*, in: arXiv preprint arXiv:1908.03295, 2019, [[OpenAccess](https://arxiv.org/abs/1908.03295)], [[Pytorch](https://github.com/zxhr2793/PADet)], `PADet`
6. **Cascade RetinaNet: Maintaining Consistency for Single-Stage Object Detection**, *Q. Tang, S. Liu, J. Li, Y. Hu*, in: BMVC, 2019, [[OpenAccess](https://arxiv.org/abs/1907.06881)], `CaRetinaNet`



## 2. Face Detection

1. **Joint face detection and alignment using multi-task cascaded convolutional networks**, *K. Zhang, Z. Zhang, Z. Li, Y. Qiao*， in: IEEE Signal Processing Letters, 2016. [[OpenAccess](https://arxiv.org/abs/1604.02878)], [[Caffe](https://github.com/kuaikuaikim/MTCNN-1)], `MTCNN`
2. **Detecting faces using region-based fully convolutional networks**, *Y. Wang, X. Ji, Z. Zhou, H. Wang, Z. Li*, in: arXiv preprint arXiv:1709.05256, 2017. [[OpenAccess](https://arxiv.org/abs/1709.05256)], `Face R-FCN`
3. **Detecting faces using inside cascaded contextual cnn**,  *K. Zhang, Z. Zhang, H. Wang, Z. Li, Y. Qiao, W. Liu*, in: ICCV, 2017. [[OpenAccess](https://ai.tencent.com/ailab/media/publications/Detecting_Faces_Using_Inside_Cascaded_Contextual_CNN.pdf)]
4. **Cms-rcnn: Contextual multiscale region-based cnn for unconstrained face detection**, *C. Zhu, Y. Zheng, K. Luu, M. Savvides*, in: Deep Learning for Biometrics, 2017. [[OpenAccess](https://arxiv.org/abs/1606.05413)], `CMS-RCNN`
5. **Face r-cnn**, *H. Wang, Z. Li, X. Ji, Y. Wang*, in: arXiv preprint arXiv:1706.01061, 2017. [[OpenAccess](https://arxiv.org/abs/1706.01061)], `Face R-CNN`
6. **Scale-aware face detection**, *Z. Hao, Y. Liu, H. Qin, J. Yan, X. Li, X. Hu*, in: CVPR, 2017. [[OpenAccess](https://arxiv.org/abs/1706.09876)]
7. **Ssh: Single stage headless face detector**, *M. Najibi, P. Samangouei, R. Chellappa, L. Davis*, in: ICCV, 2017. [[OpenAccess](https://arxiv.org/abs/1708.03979)], [[Caffe](https://github.com/mahyarnajibi/SSH)], `SSH`
8. **Feature agglomeration networks for single stage face detection**, *J. Zhang, X. Wu, J. Zhu, S. C. Hoi,* in: arXiv preprint arXiv:1712.00721, 2017. [[OpenAccess](https://arxiv.org/abs/1712.00721)], `FANet`
9. **Finding tiny faces**, *P. Hu, D. Ramanan*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_Finding_Tiny_Faces_CVPR_2017_paper.pdf)], [[MatConvNet](https://github.com/peiyunh/tiny)], `S3FD`
10. **S3fd: Single shot scale-invariant face detector**, *S. Zhang, X. Zhu, Z. Lei, H. Shi, X. Wang, S. Z. Li*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_S3FD_Single_Shot_ICCV_2017_paper.pdf)], [[Caffe](https://github.com/sfzhang15/SFD)], `S3FD`
11. **Recurrent scale approximation for object detection in cnn**, *Y. Liu, H. Li, J. Yan, F. Wei, X. Wang, X. Tang*, [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Recurrent_Scale_Approximation_ICCV_2017_paper.pdf)], [[Caffe](https://github.com/sciencefans/RSA-for-object-detection)], `RSA`
12. **Anchor cascade for efficient face detection**, *B. Yu, D. Tao*, in: arXiv preprint arXiv:1805.03363, 2018. [[OpenAccess](https://arxiv.org/abs/1805.03363)]
13. **Face detection using improved faster rcnn**, *C. Zhang, X. Xu, D. Tu*, in: arXiv preprint arXiv:1802.02142, 2018. [[OpenAccess](https://arxiv.org/abs/1802.02142)], [[Caffe](https://github.com/playerkk/face-py-faster-rcnn)]
14. **Face-magnet: Magnifying feature maps to detect small faces**, *P. Samangouei, M. Najibi, L. Davis, R. Chellappa*, in: arXiv preprint arXiv:1803.05258, 2018. [[OpenAccess](https://arxiv.org/abs/1803.05258)], [[Caffe](https://github.com/po0ya/face-magnet)]
15. **Selective refinement network for high performance face detection**, *C. Chi, S. Zhang, J. Xing, Z. Lei, S. Z. Li, X. Zou*, in: arXiv preprint arXiv:1809.02693, 2018. [[OpenAccess](https://arxiv.org/abs/1809.02693)], [[Pytorch](https://github.com/ChiCheng123/SRN)], `SRN`
16. **Pyramidbox: A context-assisted single shot face detector**, *X. Tang, D. K. Du, Z. He, J. Liu*, in: ECCV, 2018. [[OpenAccess](https://arxiv.org/abs/1803.07737)], [[TensorFlow](https://github.com/EricZgw/PyramidBox)]
17. **Face detection using deep learning: An improved faster rcnn approach**, *X. Sun, P. Wu, S. C. Hoi*, in: Neurocomputing, 2018. [[OpenAccess](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5000&context=sis_research)]
18. **Seeing small faces from robust anchors perspective**, *C. Zhu, R. Tao, K. Luu, M. Savvides*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3468.pdf)]
19. **Dsfd: Dual shot face detector**, *J. Li, Y. Wang, C. Wang, Y. Tai, J. Qian, J. Yang, C. Wang, J. Li, F. Huang*, in: CVPR, 2019. [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf)], [[Pytorch](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)], `DSFD`


## 3. Pedestrian Detection

1. **Bilattice-based logical reasoning for human detection**, *V. D. Shet, J. Neumann, V. Ramesh, L. S. Davis*, in: CVPR, 2007. [[OpenAccess](https://ieeexplore.ieee.org/document/4270158)]
2. **Integral channel features**, *P. Dollar, Z. Tu, P. Perona, S. Belongie*, in: BMVC, 2009. [[OpenAccess](https://pages.ucsd.edu/~ztu/publication/dollarBMVC09ChnFtrs_0.pdf)], [[Project](https://pdollar.github.io/toolbox/)], `ICF`
3. **A structural filter approach to human detection**, *G. Duan, H. Ai, S. Lao*, in: ECCV, 2010. [[OpenAccess](https://link.springer.com/chapter/10.1007/978-3-642-15567-3_18)]
4. **Multi-cue pedestrian classification with partial occlusion handling**, *M. Enzweiler, A. Eigenstetter, B. Schiele, D. M. Gavrila*, in: CVPR, 2010. [[OpenAccess](http://gavrila.net/Publications/cvpr10_occlusion.pdf)]
5. **A discriminative deep model for pedestrian detection with occlusion handling**, *W. Ouyang, X. Wang*, in: CVPR, 2012. [[OpenAccess](http://islab.ulsan.ac.kr/files/announcement/437/A%20Discriminative%20Deep%20Model%20for%20Pedestrian%20Detection%20with%20Occlusion%20Handling.pdf)]
6. **Modeling mutual visibility relationship in pedestrian detection**, *W. Ouyang, X. Zeng, X. Wang*, in: CVPR, 2013. [[OpenAccess](http://www.ee.cuhk.edu.hk/~xgwang/papers/ouyangZWcvpr13.pdf)]
7. **Single-pedestrian detection aided by multi-pedestrian detection**, *W. Ouyang, X. Wang*, in: CVPR, 2013. [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ouyang_Single-Pedestrian_Detection_Aided_2013_CVPR_paper.pdf)]
8. **Pedestrian detection with unsupervised multi-stage feature learning**, *P. Sermanet, K. Kavukcuoglu, S. Chintala, Y. LeCun*, in: CVPR, 2013. [[OpenAccess](https://arxiv.org/abs/1212.0142)]
9. **Joint deep learning for pedestrian detection**, *W. Ouyang, X. Wang*, in: ICCV, 2013. [[OpenAccess](http://www.ee.cuhk.edu.hk/~xgwang/papers/ouyangWiccv13.pdf)]
10. **Handling occlusions with franken-classifiers**, *M. Mathias, R. Benenson, R. Timofte, L. Van Gool*, in: ICCV, 2013. [[OpenAccess](http://rodrigob.github.io/documents/2013_iccv_occlusions_with_supplementary_material.pdf)]
11. **Ten years of pedestrian detection, what have we learned?**, *R. Benenson, M. Omran, J. Hosang, B. Schiele*, in: ECCV, 2014. [[OpenAccess](https://rodrigob.github.io/documents/2014_eccvw_ten_years_of_pedestrian_detection_with_supplementary_material.pdf)]
12. **Detection and tracking of occluded people**, *S. Tang, M. Andriluka, B. Schiele*, in: IJCV, 2014. [[OpenAccess](https://dl.acm.org/citation.cfm?id=2674557)]
13. **Learning complexity-aware cascades for deep pedestrian detection**, *Z. Cai, M. Saberian, N. Vasconcelos*, in: ICCV, 2015. [[OpenAccess](http://openaccess.thecvf.com/content_iccv_2015/papers/Cai_Learning_Complexity-Aware_Cascades_ICCV_2015_paper.pdf)]
14. **Taking a deeper look at pedestrians**, *J. Hosang, M. Omran, R. Benenson, B. Schiele*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hosang_Taking_a_Deeper_2015_CVPR_paper.pdf)]
15. **Deep learning strong parts for pedestrian detection**, *Y. Tian, P. Luo, X. Wang, X. Tang*, in: CVPR, 2015. [[OpenAccess](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf)]
16. **A unified multi-scale deep convolutional neural network for fast object detection**, *Z. Cai, Q. Fan, R. S. Feris, N. Vasconcelos*, [[OpenAccess](http://www.eccv2016.org/files/posters/P-2B-38.pdf)], [[Caffe](https://github.com/zhaoweicai/mscnn)], `MSCNN`
17. **Dave: A unified framework for fast vehicle detection and annotation**, *Y. Zhou, L. Liu, L. Shao, M. Mellor*, in: ECCV, 2016. [[OpenAccess](https://arxiv.org/abs/1607.04564)]
18. **Is faster r-cnn doing well for pedestrian detection?**, *L. Zhang, L. Lin, X. Liang, K. He*, in: ECCV, 2016. [[OpenAccess](http://kaiminghe.com/publications/eccv16ped.pdf)], [[Caffe](https://github.com/zhangliliang/RPN_BF)]
19. **Exploit all the layers: Fast and accurate cnn object detector with scale dependent pooling and cascaded rejection classifiers**, *F. Yang, W. Choi, Y. Lin*, [ [OpenAccess](http://openaccess.thecvf.com/content_cvpr_2016/html/Yang_Exploit_All_the_CVPR_2016_paper.html)], `SDP-CRC`
20. **Accurate single stage detector using recurrent rolling convolution**, *J. Ren, X. Chen, J. Liu, W. Sun, J. Pang, Q. Yan, Y.-W. Tai, L. Xu*, [[OpenAccess](http://zpascal.net/cvpr2017/Ren_Accurate_Single_Stage_CVPR_2017_paper.pdf)], [[Caffe](https://github.com/xiaohaoChen/rrc_detection)], `RRC`
21. **What can help pedestrian detection?**, *J. Mao, T. Xiao, Y. Jiang, Z. Cao*, in: CVPR, 2017. [[OpenAccess](http://zpascal.net/cvpr2017/Mao_What_Can_Help_CVPR_2017_paper.pdf)]
22. **Learning cross-modal deep representations for robust pedestrian detection**, *D. Xu, W. Ouyang, E. Ricci, X. Wang, N. Sebe*, in: CVPR, 2017. [[OpenAccess](https://arxiv.org/abs/1704.02431)], [[Caffe](https://github.com/danxuhk/CMT-CNN)], `CMT-CNN`
23. **Repulsion loss: Detecting pedestrians in a crowd**, *X. Wang, T. Xiao, Y. Jiang, S. Shao, J. Sun, C. Shen*, in: CVPR, 2018. [[OpenAccess](http://zpascal.net/cvpr2018/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)], [[Pytorch](https://github.com/bailvwangzi/repulsion_loss_ssd)]
24. **Bi-box regression for pedestrian detection and occlusion estimation**, *C. Zhou, J. Yuan*, in: ECCV, 2018. [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.html)]
25. **Occlusion-aware r-cnn: Detecting pedestrians in a crowd**, *S. Zhang, L. Wen, X. Bian, Z. Lei, S. Z. Li*, in: ECCV, 2018. [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)], `OR R-CNN`
26. **Scale-aware fast r-cnn for pedestrian detection**, *J. Li, X. Liang, S. Shen, T. Xu, J. Feng, S. Yan*,  [[Arxiv](https://arxiv.org/pdf/1510.08160)], in: TMM, 2018. `SAF R-CNN`
27. **Pcn: Part and context information for pedestrian detection with cnns**, *S. Wang, J. Cheng, H. Liu, M. Tang*, in: arXiv preprint arXiv:1804.04483, 2018. [[OpenAccess](https://arxiv.org/abs/1804.04483)]


## 4 Benchmarks

### 4.1 Generic Detection Datasets

1. `Pascal VOC`: **The pascal visual object classes (voc) challenge**, *M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman*, [[OpenAccess](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)], [[Project](http://host.robots.ox.ac.uk/pascal/VOC/)]
2. `ImageNet`: **Imagenet: A large-scale hierarchical image database**, * J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei*, [[OpenAccess](http://www.image-net.org/papers/imagenet_cvpr09.pdf)], [[Project](http://www.image-net.org/)]
3. `MSCOCO`: **Microsoft COCO: Common Objects in Context**, *T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, C. L. Zitnick*, [[OpenAccess](https://arxiv.org/pdf/1405.0312)], [[Project](http://cocodataset.org/)]
4. `Open Images`: **The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale**, *A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, T. Duerig, et al.*, [[OpenAccess](https://arxiv.org/abs/1811.00982)], [[Project](https://opensource.google.com/projects/open-images-dataset)]
5. `LVIS`: **Lvis: A dataset for large vocabulary instance segmentation**, *A. Gupta, P. Dollar, R. Girshick*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf)], [[Project](https://www.lvisdataset.org/)]

### 4.2 Face Detection Datasets

1. `WIDER FACE`: **Wider face: A face detection benchmark**, *S. Yang, P. Luo, C.-C. Loy, X. Tang*, [[OpenAccess](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_WIDER_FACE_A_CVPR_2016_paper.pdf)], [[Project](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace)]
2. `FDDB`: **Fddb: A benchmark for face detection in unconstrained settings**, *V. Jain, E. Learned-Miller*, [[OpenAccess](http://vis-www.cs.umass.edu/fddb/fddb.pdf)], [[Project](http://vis-www.cs.umass.edu/fddb/)]
3. `PASCAL FACE`: **The pascal visual object classes (voc) challenge**, *M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman*, [[OpenAccess](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)], [[Project](http://host.robots.ox.ac.uk/pascal/VOC/)]
4. `MALF`: **Automatic Face and Gesture Recognition (FG)**, *Yang, Bin and Yan, Junjie and Lei, Zhen and Li, Stan Z*, [[OpenAccess](http://www.cbsr.ia.ac.cn/faceevaluation/faceevaluation15.pdf)], [[Project](http://www.cbsr.ia.ac.cn/faceevaluation/)]
5. `AFW`: **Face detection, pose estimation and landmark localization in the wild**, *X. Zhu, D. Ramanan*, [[OpenAccess](http://www.cs.cmu.edu/~deva/papers/face/face-cvpr12.pdf)], [[Project](http://www.cs.cmu.edu/~deva/papers/face/index.html)]

### 4.3 Pedestrian Detection Datasets

1. `CityPersons`: **Citypersons: A diverse dataset for pedestrian detection**, *S. Zhang, R. Benenson, B. Schiele*, [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Zhang_CityPersons_A_Diverse_2017_CVPR_supplemental.pdf)], [[Project](https://www.cityscapes-dataset.com)]
2. `Caltech`: **Pedestrian detection: An evaluation of the state of the art**, *P. Dollar, C. Wojek, B. Schiele, P. Perona*, [[OpenAccess](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.6884&rep=rep1&type=pdf)], [[Project](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)]
3. `ETH`: **Depth and appearance for mobile scene analysis**, *A. Ess, B. Leibe, L. Van Gool*, [[OpenAccess](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.212.8331&rep=rep1&type=pdf)], [[Project](https://data.vision.ee.ethz.ch/cvl/aess/iccv2007/)]
4. `INRIA`: **Histograms of oriented gradients for human detection**, *N. Dalal, B. Triggs*, [[OpenAccess](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)], [[Project](http://pascal.inrialpes.fr/data/human/)]
5. `KITTI`: **Vision meets robotics: The kitti dataset**, *A. Geiger, P. Lenz, C. Stiller, R. Urtasun*, [[OpenAccess](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)], [[Project](http://www.cvlibs.net/datasets/kitti/)]


## 5. SOTA

### 5.1 Pascal VOC

|Method| Backbone| Proposed Year| Input size(Test) | VOC2007 |VOC2012|
|---| ---| ---| --- |---| --- |
|R-CNN | VGG-16| 2014| Arbitrary| 66.0∗ |62.4†|
|SPP-net | VGG-16 |2014 |~600 × 1000| 63.1∗| -|
|Fast R-CNN|VGG-16 |2015 ∼ |600 × 1000| 70.0 |68.4|
|Faster R-CNN | VGG-16| 2015 |~600 × 1000 |73.2 |70.4|
|MR-CNN | VGG-16 |2015| Multi-Scale| 78.2 |73.9|
|Faster R-CNN | ResNet-101 |2016 |~600 × 1000 |76.4| 73.8|
|R-FCN| ResNet-101| 2016| ~600 × 1000| 80.5| 77.6|
|OHEM| VGG-16| 2016| ~600 × 1000|74.6 |71.9|
|HyperNet | VGG-16| 2016| ~600 × 1000| 76.3| 71.4|
|ION | VGG-16| 2016| ~600 × 1000| 79.2| 76.4|
|CRAFT |VGG-16 |2016| ~600 × 1000| 75.7 |71.3†
|LocNet | VGG-16 |2016| ~600 × 1000 |78.4 |74.8†|
|R-FCN w DCN | ResNet-101 |2017 |~600 × 1000 |82.6| -|
|CoupleNet | ResNet-101| 2017 |~600 × 1000| 82.7| 80.4|
|DeNet512(wide) | ResNet-101| 2017| ~512 × 512 |77.1| 73.9|
|FPN-Reconfig | ResNet-101| 2018| ~600 × 1000| 82.4| 81.1|
|DeepRegionLet| ResNet-101 |2018| ~600 × 1000| 83.3| 81.3|
|DCN+R-CNN | ResNet-101+ResNet-152| 2018| Arbitrary| 84.0 |81.2|
|YOLOv1| VGG16 |2016 |448 × 448| 66.4| 57.9|
|SSD512| VGG-16 |2016 |512 × 512| 79.8| 78.5|
|YOLOv2 | Darknet 2017| 544 × 544| 78.6| 73.5|
|DSSD513| ResNet-101| 2017 |513 × 513| 81.5| 80.0|
|DSOD300 | DS/64-192-48-1| 2017 |300 × 300| 77.7 |76.3|
|RON384 | VGG-16| 2017 |384 × 384| 75.4| 73.0|
|STDN513 | DenseNet-169| 2018 |513 × 513 |80.9| -|
|RefineDet512 | VGG-16 |2018 |512 × 512 |81.8 |80.1|
|RFBNet512  |VGG16 |2018 |512 × 512| 82.2| -|
|CenterNet | ResNet101 |2019| 512 × 512 |78.7| -|
|CenterNet | DLA | 2019| 512 × 512| 80.7 |-|

*∗: This entry reports the the model is trained with VOC2007 trainval sets only.*
*†: This entry reports the the model are trained with VOC2012 trainval sets only .*

### 5.2 MSCOCO

|Method |Backbone |Year |AP |AP$_{50}$ |AP$_{75}$ |AP$_{S}$ |AP$_{M}$ |AP$_{L}$|
|--- |--- |--- |--- |--- |--- |--- |--- |---|
|Fast R-CNN | VGG-16 |2015 |19.7 |35.9| - |- |- |-|
|Faster R-CNN | VGG-16| 2015 |21.9| 42.7| - |-| -| -|
|OHEM | VGG-16 |2016 |22.6| 42.5| 22.2 |5.0| 23.7| 37.9|
|ION | VGG-16 |2016 |23.6 |43.2| 23.6| 6.4| 24.1 |38.3|
|OHEM++| VGG-16| 2016| 25.5 |45.9 |26.1 |7.4| 27.7| 40.3|
|R-FCN| ResNet-101| 2016 |29.9| 51.9| - |10.8| 32.8| 45.0|
|Faster R-CNN+++ | ResNet-101| 2016| 34.9 |55.7 |37.4|15.6| 38.7| 50.9|
|Faster R-CNN w FPN | ResNet-101 |2016 |36.2| 59.1| 39.0| 18.2| 39.0 |48.2|
|DeNet-101(wide) | ResNet-101| 2017| 33.8| 53.4| 36.1 |12.3| 36.1|50.8|
|CoupleNet | ResNet-101 |2017| 34.4 |54.8| 37.2 |13.4| 38.1| 50.8|
|Faster R-CNN by G-RMI | Inception-ResNet-v2| 2017| 34.7| 55.5| 36.7 |13.5 |38.1 |52.0|
|Deformable R-FCN| Aligned-Inception-ResNet| 2017| 37.5 |58.0| 40.8 |19.4 |40.1 |52.5|
|Mask-RCNN | ResNeXt-101| 2017| 39.8| 62.3 |43.4| 22.1| 43.2| 51.2|
|umd det | ResNet-101| 2017| 40.8 |62.4 |44.9| 23.0 |43.4| 53.2|
|Fitness-NMS | ResNet-101 |2017| 41.8| 60.9 |44.9| 21.5 |45.0| 57.5|
|DCN w Relation Net | ResNet-101 |2018 |39.0| 58.6| 42.9| -| - |-|
|DeepRegionlets | ResNet-101 |2018 |39.3| 59.8 |- |21.7| 43.7| 50.9|
|C-Mask RCNN | ResNet-101 |2018| 42.0| 62.9 |46.4| 23.4 |44.7| 53.8|
|Group Norm| ResNet-101 |2018|42.3| 62.8| 46.2| -| -| -|
|DCN+R-CNN | ResNet-101+ResNet-152| 2018 |42.6| 65.3 |46.5| 26.4| 46.1 |56.4|
|Cascade R-CNN | ResNet-101| 2018 |42.8| 62.1| 46.3| 23.7| 45.5| 55.2|
|SNIP++ | DPN-98| 2018 |45.7| 67.3| 51.1 |29.3 |48.8 |57.1|
|SNIPER++ | ResNet-101| 2018 |46.1 |67.0| 51.6 |29.6 |48.9| 58.1|
|PANet++ | ResNeXt-101 |2018| 47.4| 67.2| 51.8 |30.1 |51.7 |60.0|
|Grid R-CNN | ResNeXt-101| 2019 |43.2| 63.0| 46.6 |25.1 |46.5| 55.2|
|DCN-v2 | ResNet-101 |2019| 44.8 |66.3 |48.8 |24.4 |48.1 |59.6|
|DCN-v2++| ResNet-101| 2019 |46.0| 67.9| 50.8 |27.8| 49.1 |59.5|
|TridentNet| ResNet-101| 2019| 42.7 |63.6 |46.5| 23.9| 46.6| 56.6|
|TridentNet | ResNet-101-Deformable |2019 |48.4 |69.7| 53.5| 31.8| 51.3| 60.3|
|SSD512 | VGG-16 |2016| 28.8 |48.5| 30.3| 10.9| 31.8 |43.5|
|RON384++| VGG-16 |2017| 27.4| 49.5| 27.1| -| -| -|
|YOLOv2 | DarkNet-19 |2017| 21.6| 44.0| 19.2| 5.0| 22.4 |35.5|
|SSD513| ResNet-101| 2017 |31.2| 50.4 |33.3| 10.2 |34.5 |49.8|
|DSSD513 | ResNet-101| 2017 |33.2| 53.3 |35.2| 13.0| 35.4 |51.1|
|RetinaNet800++ | ResNet-101 |2017 |39.1 |59.1 |42.3| 21.8| 42.7| 50.2|
|STDN513 | DenseNet-169| 2018| 31.8| 51.0| 33.6| 14.4| 36.1| 43.4|
|FPN-Reconfig | ResNet-101| 2018 |34.6 |54.3 |37.3| -| - |-|
|RefineDet512 | ResNet-101| 2018 |36.4 |57.5 |39.5 |16.6| 39.9| 51.4|
|RefineDet512++ | ResNet-101 |2018 |41.8| 62.9| 45.7| 25.6| 45.1| 54.1|
|GHM SSD | ResNeXt-101| 2018 |41.6| 62.8| 44.2| 22.3 |45.1| 55.3|
|CornerNet511 | Hourglass-104| 2018| 40.5 |56.5| 43.1| 19.4| 42.7| 53.9|
|CornerNet511++ | Hourglass-104 |2018 |42.1 |57.8 |45.3| 20.8| 44.8| 56.7|
|M2Det800 | VGG-16 |2019| 41.0| 59.7 |45.0 |22.1 |46.5| 53.8|
|M2Det800++ | VGG-16 |2019| 44.2 |64.6 |49.3 |29.2 |47.9| 55.1|
|ExtremeNet| Hourglass-104| 2019| 40.2 |55.5 |43.2| 20.4| 43.2| 53.1|
|CenterNet-HG| Hourglass-104| 2019| 42.1 |61.1| 45.9| 24.1| 45.5| 52.8|
|FCOS | ResNeXt-101| 2019 |42.1| 62.1 |45.2| 25.6| 44.9| 52.0|
|FSAF | ResNeXt-101 |2019 |42.9| 63.8 |46.3 |26.6| 46.2| 52.7|
|CenterNet511 | Hourglass-104 |2019 |44.9| 62.4 |48.1| 25.6| 47.4 |57.4|
|CenterNet511++ |Hourglass-104 |2019| 47.0| 64.5| 50.7| 28.9| 49.9| 58.9|




## 6. Future Work

### 6.1 Anchor Design

#### 6.1.1 Anchor-Free Methods

1. **Denet: Scalable real-time object detection with directed sparse sampling**, *L. Tychsen-Smith, L. Petersson*, in: ICCV, 2017. [[OpenAccess](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tychsen-Smith_DeNet_Scalable_Real-Time_ICCV_2017_paper.pdf)],[[Theano](https://github.com/lachlants/denet)], `DeNet`
2. **Cornernet: Detecting objects as paired keypoints**, *H. Law, J. Deng*, in: ECCV, 2018. [[OpenAccess](http://openaccess.thecvf.com/content_ECCV_2018/html/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.html)], [[Pytorch](https://github.com/princeton-vl/CornerNet)], `CornerNet`
3. **Objects as points**, *X. Zhou, D. Wang, P. Krahenb ¨ uhl*, [[Arxiv](https://arxiv.org/pdf/1904.07850)], [[Pytorch](https://github.com/xingyizhou/CenterNet)],  in: arXiv preprint arXiv:1904.07850, 2019. `CenterNet`
4. **Centernet: Keypoint triplets for object detection**, *K. Duan, S. Bai, L. Xie, H. Qi, Q. Huang, Q. Tian*, in: arXiv preprint arXiv:1904.08189, 2019. [[Arxiv](https://arxiv.org/pdf/1904.08189)], [[Pytorch](https://github.com/Duankaiwen/CenterNet)], `CenterNet`
5. **Bottom-up object detection by grouping extreme and center points**, *X. Zhou, J. Zhuo, P. Krahenbuhl*, in: CVPR, 2019. [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_Bottom-Up_Object_Detection_by_Grouping_Extreme_and_Center_Points_CVPR_2019_paper.pdf)], [[Pytorch](https://github.com/xingyizhou/ExtremeNet)], `ExtremeNet`
6. **Feature selective anchor-free module for single-shot object detection**, *C. Zhu, Y. He, M. Savvides*, in: CVPR, 2019. [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.pdf)], `FSFA`
7. **Fcos: Fully convolutional one-stage object detection**, *Z. Tian, C. Shen, H. Chen, T. He*, in: ICCV, 2019. [[OpenAccess](https://arxiv.org/abs/1904.01355)], [[Pytorch](https://github.com/tianzhi0549/FCOS)], `FCOS`
8. **CornerNet-Lite: Efficient Keypoint Based Object Detection**, *Hei Law, Yun Teng, Olga Russakovsky, Jia Deng*, in: arXiv preprint arXiv:1904.08900, 2019. [[OpenAccess](https://arxiv.org/abs/1904.08900)], [[Pytorch](https://github.com/princeton-vl/CornerNet-Lite)], `CornerNet-Lite`
9. **RepPoints: Point Set Representation for Object Detection**, *Z. Yang, S. Liu, H. Hu, L. Wang, S. Lin*, in: ICCV, 2019. [[OpenAccess](https://arxiv.org/abs/1904.11490)], `RepPoints`

#### 6.1.2 Anchor-Refinement Methods

1. **Yolo9000: better, faster, stronger**, *J. Redmon, A. Farhadi*, [[OpenAccess](http://zpascal.net/cvpr2017/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)], [[DarkNet](https://pjreddie.com/darknet/yolo)], in: CVPR, 2017. `YOLOv2`
2. **Cascade r-cnn: Delving into high quality object detection**, *Z. Cai, N. Vasconcelos*, in: CVPR, 2018. [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/zhaoweicai/cascade-rcnn)], [[Caffe2](https://github.com/zhaoweicai/Detectron-Cascade-RCNN)] `Cascade R-CNN`
3. **Single-shot refinement neural network for object detection**, *S. Zhang, L. Wen, X. Bian, Z. Lei, S. Z. Li*, in: CVPR, 2018. [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf)], [[Caffe](https://github.com/sfzhang15/RefineDet)], `RefineDet`
4. **Metaanchor: Learning to detect objects with customized anchors**, *T. Yang, X. Zhang, Z. Li, W. Zhang, J. Sun*, in: NeurIPS, 2018.  [[OpenAccess](https://papers.nips.cc/paper/7315-metaanchor-learning-to-detect-objects-with-customized-anchors)], `MetaAnchor`
5. **Derpn: Taking a further step toward more general object detection**, *L. J. Z. X. Lele Xie, Yuliang Liu*, in: AAAI, 2019. [[OpenAccess](https://www.aaai.org/ojs/index.php/AAAI/article/view/4936/4809)], [[Caffe](https://github.com/HCIILAB/DeRPN)], `DeRPN`
6. **Region Proposal by Guided Anchoring**, *J. Wang, K. Chen, S. Yang, C. C. Loy, D. Lin*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Region_Proposal_by_Guided_Anchoring_CVPR_2019_paper.pdf)], [[mmdetection](https://github.com/open-mmlab/mmdetection)]
7. **Revisiting Feature Alignment for One-stage Object Detection**, *Y. Chen, C. Han, N. Wang, Z. Zhang*, in: arXiv preprint arXiv:1908.01570, 2019, [[OpenAccess](https://arxiv.org/abs/1908.01570)], `AlignDet`
8. **PosNeg-Balanced Anchors with Aligned Features for Single-Shot Object Detection**, *Qiankun Tang, Shice Liu, Jie Li, Yu Hu*, in: arXiv preprint arXiv:1908.03295, 2019, [[OpenAccess](https://arxiv.org/abs/1908.03295)], [[Pytorch](https://github.com/zxhr2793/PADet)], `PADet`
9. **Cascade RetinaNet: Maintaining Consistency for Single-Stage Object Detection**, *Q. Tang, S. Liu, J. Li, Y. Hu*, in: BMVC, 2019, [[OpenAccess](https://arxiv.org/abs/1907.06881)], `CaRetinaNet`



### 6.2 AutoML Detection

1. **Nas-fpn: Learning scalable feature pyramid architecture for object detection**, *G. Ghiasi, T.-Y. Lin, Q. V. Le*, [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ghiasi_NAS-FPN_Learning_Scalable_Feature_Pyramid_Architecture_for_Object_Detection_CVPR_2019_paper.pdf)], [[TensorFlow](https://github.com/DetectionTeamUCAS/NAS_FPN_Tensorflow)], `NAS-FPN`
2. **Detnas: Neural architecture search on object detection**, *Y. Chen, T. Yang, X. Zhang, G. Meng, C. Pan, J. Sun*, in: arXiv preprint arXiv:1903.10979, 2019. [[OpenAccess](https://arxiv.org/abs/1903.10979)], `DetNas`
3. **Learning data augmentation strategy**, *B. Zoph, E. D. Cubuk, G. Ghiasi, T.-Y. Lin, J. Shlens, Q. V. Le*,  in: arXiv preprint arXiv:1906.11172, 2019. [[OpenAccess](https://arxiv.org/abs/1906.11172)], [[TensorFlow](https://github.com/tensorflow/tpu/tree/master/models/official/detection)]
4. **AutoAugment: Learning Augmentation Strategies from Data**, *E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, Q. V. Le*,  in: CVPR, 2019. [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf)], `AutoAugment`

### 6.3 Low-shot Detection

1. **Few-example object detection with model communication**, *X. Dong, L. Zheng, F. Ma, Y. Yang, D. Meng*, in: TPAMI, 2018. [[OpenAccess](https://arxiv.org/abs/1706.08249)], [[Project](https://github.com/D-X-Y/DXY-Projects)], `MSPLD`
2. **Lstd: A low-shot transfer detector for object detection**, *H. Chen, Y. Wang, G. Wang, Y. Qiao*, in: AAAI, 2018.
 [[OpenAccess](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16778/16580)], [[Caffe](https://github.com/Cassie94/LSTD)], `LSTD`
3. **Repmet: Representative-based metric learning for classification and one-shot object detection**, *E. Schwartz, L. Karlinsky, J. Shtok, S. Harary, M. Marder, S. Pankanti, R. Feris, A. Kumar, R. Giries, A. M. Bronstein*,  in: CVPR, 2019. [[OpenAccess](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karlinsky_RepMet_Representative-Based_Metric_Learning_for_Classification_and_Few-Shot_Object_Detection_CVPR_2019_paper.pdf)], [[Pytorch](https://github.com/HaydenFaulkner/pytorch.repmet)], `RepMet`


### 6.4 Others

1. **Megdet: A large mini-batch object detector**, *C. Peng, T. Xiao, Z. Li, Y. Jiang, X. Zhang, K. Jia, G. Yu, J. Sun*, in: CVPR, 2018 [[OpenAccess](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0836.pdf)], `Megdet`
2. **Incremental learning of object detectors without catastrophic forgetting**, *K. Shmelkov, C. Schmid, K. Alahari, *, in: ICCV, 2017. [[OpenAccess](https://arxiv.org/abs/1708.06977)], [[TensorFlow](https://github.com/kshmelkov/incremental_detectors)]


## 7. Other Resources

- [Object Detection](https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html)
- [hoya012/deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)
- [amusi/awesome-object-detection](https://github.com/amusi/awesome-object-detection)


