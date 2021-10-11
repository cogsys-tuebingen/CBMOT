# Score refinement for confidence-based 3D multi-object tracking

Our [VIDEO](https://youtu.be/0GL_lsUvPdo) gives a brief explanation of our Method.

This is the official code for the paper:
> [**Score refinement for confidence-based 3D multi-object tracking**](https://arxiv.org/abs/2107.04327),            
> Nuri Benbarka, Jona Schröder, Andreas Zell,        
> *arXiv technical report ([arXiv 2107.04327](https://arxiv.org/abs/2107.04327))*  

    @article{benbarka2021score,
        title={Score refinement for confidence-based 3D multi-object tracking},
        author={Benbarka, Nuri and Schr{\"o}der, Jona and Zell, Andreas},
        journal={arXiv preprint arXiv:2107.04327},
        year={2021}
    }

It also contains the code of the B.Sc. thesis:

> [**Learning score update functions for confidence-based MOT**](https://www.researchgate.net/publication/354418608_Learning_score_update_functions_for_confidence-based_MOT),
> Anouar Gherri,
        
    @article{gherri2021learning,
        title = {Learning score update functions for confidence-based MOT},
        author = {Gherri, Anouar},
        year = {2021}        
    }

## Contact
Feel free to contact us for any questions! 

Nuri Benbarka [nuri.benbarka@uni-tuebingen.de](mailto:nuri.benbarka@uni-tuebingen.de),

Jona Schröder [jona.schroeder@uni-tuebingen.de](mailto:jona.schroeder@uni-tuebingen.de),

Anouar Gherri [anouargherri93@gmail.com](mailto:anouargherri93@gmail.com), 

## Abstract
Multi-object tracking is a critical component in autonomous navigation, as it provides valuable information for decision-making. Many researchers tackled the 3D multi-object tracking task by filtering out the frame-by-frame 3D detections; however, their focus was mainly on finding useful features or proper matching metrics. Our work focuses on a neglected part of the tracking system: score refinement and tracklet termination. We show that manipulating the scores depending on time consistency while terminating the tracklets depending on the tracklet score improves tracking results. We do this by increasing the matched tracklets' score with score update functions and decreasing the unmatched tracklets' score. Compared to count-based methods, our method consistently produces better AMOTA and MOTA scores when utilizing various detectors and filtering algorithms on different datasets. The improvements in AMOTA score went up to 1.83 and 2.96 in MOTA. We also used our method as a late-fusion ensembling method, and it performed better than voting-based ensemble methods by a solid margin. It achieved an AMOTA score of 67.6 on nuScenes test evaluation, which is comparable to other state-of-the-art trackers. 

## Results

#### NuScenes

|  Detector  |  Split  |  Update function  |  modality  |  AMOTA  |  AMOTP  |  MOTA  |
|---------|---------|---------|--------|-------|--------|--------|
| CenterPoint  |  Val    |  -  |  Lidar  |  67.3 |  57.4 |  57.3 |
| CenterTrack  |  Val    |  -  |  Camera  |  17.8 |  158.0  |  15.0  |
| CenterPoint  |  Val    |  Multiplication  |  Lidar  |  68.8 |  58.9 |  60.2 |
| CenterPoint + CenterTrack  |  Val   |   Multiplication  |  Fusion  | 72.1 |  53.3 |  58.5 |
| CenterPoint + CenterTrack  |  Val   |   Neural network  |  Fusion  | 72.0 |  48.7 |  58.2 |

The results are different than what is reported in the paper because of optimizing NUSCENE_CLS_VELOCITY_ERRORs, 
and using the new detection results from CenterPoint.

## Installation

```bash
# basic python libraries
conda create --name CBMOT python=3.7
conda activate CBMOT
git clone https://github.com/cogsys-tuebingen/CBMOT.git
cd CBMOT
pip install -r requirements.txt
```
Create a folder to place the dataset called data. 
Download the NuScenes dataset and then prepare it as was instructed in [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup). 
Make a hyperlink that points to the prepared dataset. 
```bash
mkdir data
cd data
ln -s  LINK_TO_NUSCENES_DATA_SET ./nuScenes
cd ..
```


Ceate a folder named resources. 

```bash
mkdir resources
```
Download the detections/tracklets and place them in the resources folder.
We used CenterPoint [detections](https://drive.google.com/drive/folders/1FOfCe9nWQrySUx42PlZyaKWAK2Or0sZQ) (LIDAR) and CenterTrack tracklets (Camera). 
If you don't want to run CenterTrack yourself, we have the tracklets [here](https://u-173-c142.cs.uni-tuebingen.de/index.php/s/dMmY2iRC7azD2Lw). 
For the experiment with the learned score update function, please download the network's weights from [here](https://u-173-c142.cs.uni-tuebingen.de/index.php/s/dMmY2iRC7azD2Lw).


### Usage

We made a bash script Results.sh to get the result table above. Running the script should take approximately 4 hours.
```bash
bash Results.sh
```

### Learning update function model
In the directory learning_score_update_function 
* open lsuf_train
* put your CMOT project path into CMOT_path
* run the file to generate the model from the best results
* feel free to experiment yourself different parameters
## Acknowledgment
This project is not possible without multiple great open sourced codebases. We list some notable examples below.  

* [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [P3DMOT](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking)

**CBMOT is deeply influenced by the following projects. Please consider citing the relevant papers.**

```
@article{zhu2019classbalanced,
  title={Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection},
  author={Zhu, Benjin and Jiang, Zhengkai and Zhou, Xiangxin and Li, Zeming and Yu, Gang},
  journal={arXiv:1908.09492},
  year={2019}
}

@article{lang2019pillar,
   title={PointPillars: Fast Encoders for Object Detection From Point Clouds},
   journal={CVPR},
   author={Lang, Alex H. and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
   year={2019},
}

@inproceedings{yin2021center,
  title={Center-based 3d object detection and tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Krahenbuhl, Philipp},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11784--11793},
  year={2021}
}

@article{zhou2020tracking,
  title={Tracking Objects as Points},
  author={Zhou, Xingyi and Koltun, Vladlen and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv:2004.01177},
  year={2020}
}

@inproceedings{weng20203d,
  title={3d multi-object tracking: A baseline and new evaluation metrics},
  author={Weng, Xinshuo and Wang, Jianren and Held, David and Kitani, Kris},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10359--10366},
  year={2020},
  organization={IEEE}
}

@article{chiu2020probabilistic,
  title={Probabilistic 3D Multi-Object Tracking for Autonomous Driving},
  author={Chiu, Hsu-kuang and Prioletti, Antonio and Li, Jie and Bohg, Jeannette},
  journal={arXiv preprint arXiv:2001.05673},
  year={2020}
}

```
