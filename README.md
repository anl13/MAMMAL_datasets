# Datasets proposed by MAMMAL
This repository presents how to download and use BamaPig2D and BamaPig3D datasets proposed by [MAMMAL](https://github.com/anl13/MAMMAL_core) system. 

![img](pics/BamaPig2D.jpg)

## Download 
BamaPig2D (8.02GB for zipflie. 9.23G after unzip, yet occupy 10.7G space on windows) can be downloaded from [Google Drive](https://drive.google.com/file/d/1yWBtNpYpkUdGKDqUAE7ya5m_fwinn0HN/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1vTwipVuXHNhBFc91tNXteQ) (extract code: vj9n).

BamaPig2D_sleap (33.7M, only contains `.json` files) can be downloaded from [Google Drive](https://drive.google.com/file/d/1XRFvUM8iBtzkIzr83rMkpQ4NonTKY9dk/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1PC_n9_nqRsduw5JYuyTOVA) (extract code: qtb9). 

BamaPig3D (8.86GB for zipfile. 9.62G after unzip yet occupy 24.9G space on windows because it contains many small files) can be downloaded from [Google Drive](https://drive.google.com/file/d/1rZtuR9B9ojxQKkps0j5duwAJ-v3iM1k7/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1tOgf5icIt0GKI4zpV_TE4Q) (extract code: z8u6).

BamaPig3D_pure_pickle (481M for zipfile, 579M after unzip, yet occupy 941M space on Windows. ). [Google Drive](https://drive.google.com/file/d/17-jiZh4D8cYUNkzsZMSfjLL_5gyr-3i_/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1ZrAdLHwDDm1ZWqUpz94P7Q) (extract code: jams). This is a concise version containing only labeled images and labels. 
## Description

### BamaPig2D 
When you download `BamaPig2D.zip` and unzip it, you will get two folders: `images` and `annotations`. 
1. `images`: contains 3340 images used for training. 
2. `annotation`: contains two files. `train_pig_cocostyle.json` is for training and `eval_pig_cocostyle.json` is used for testing. Both are in COCO style, and you can read them using COCO PythonAPI (see also [pig_pose_det]. Train split contains 3008 images and 10356 instances, while eval split contains 332 images and 1148 instances. 
![dataset](pics/keypoint.jpg)
### BamaPig3D
BamaPig3D dataset contains 1750 images with 70 ones annotated. The contents in each folder are described below. The annotations here are mainly `.json` file or `.txt` file which are more friendly to MATALB or C++ users.

1. `image` folder contains uncalibrated synchronized images. It has 10 folders. Each folder contains 1750 images of a single view. The camera names are `0`, `1`, `2`, `5`, `6`, `7`, `8`, `9`, `10`, `11`. Images are in `xxxxxx.jpg` name style. 
2. `label_images` folder contains calibrated images organized the same to `image` folder. `label_images` also contains 2D annotations as `xxxxxx.json` in the same folder to `xxxxxx.jpg`. Each `xxxxxx.json` file is the output of [LabelMe software](https://pypi.org/project/labelme/) and you can use `code/demo_readBamaPig3D.py` to check how to parse these 2D information and visualize them together with uncalibrated images. The pigs are labeled in the following order: 
<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 45%;"
    src="pics/labelorder_for_3dlabel.jpg" 
    alt="">
</img>

3. `label_3d` folder contains 3D keypoints annotation. For pig `i` (`i=0,1,2,3`) and frame `k` (`k=0,25,...,1725`), the 3D keypoint file is `pig_{i}_frame_{k}.txt`. The 3D pig annotation follows the same order to 2D. 
 Each txt file is a `23*3` matrix, with the 18, 20, 22, 23 rows always set zero. Invisible keypoints without 3D labels are set to zero. Therefore, only 19 keypoints are valid which names are defined in the order. See also BamaPig2D. 

4. `label_mesh` is organized same to `label_3d`. The difference is, its keypoints totally come from the labeled mesh (i.e. the PIG model), whose pose parameters are stored in `label_pose_params`. You can use `bodymodel_np.py` and the PIG model files (see [PIG model]) to read these pose params and regress the keypoints from pose parameters. 

5. `label_mix` is organized same to `label_3d`. It is the final 3D keypoint labeling combining `label_3d` and `label_mesh`. All the experiments in the paper are performed on this labeling. Please refer to the paper for detailed decription. 

6. `boxes_pr` and `masks_pr` are detection results from [pig_silhouette_det] (a modified [PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)).

7. `keypoints_hrnet` are keypoint detection results from [pig_pose_det] (a modified HRNet) using our weights pre-trained on BamaPig2D dataset. Note that, `boxes_pr`, `masks_pr` and `keypoints_hrnet` are the detection results used to generate evaluation results in Fig.2 and the video in Supplementary Video 1 of the paper. You can test other 3D reconstruction methods fairly based on these baseline results, or just use your own detection methods to generate another detection results. 

8. `extrinsic_camera_params` contains 10 camera extrinsic paramters in `{camid}.txt` file. For example, for `00.txt`, it contains 6 float number, with the first three are camera rotation in axis-angle format, the last three are translation in xyz order. Unit is meter. `marker_3dpositions.txt` contains the 3d positions of 75 scene points for extrinsic camera parameter solving with PnP algorithm (see Supplementary Fig. 1 in the paper). `markerid_for_extrinsic_pnp.ppt` shows how these 75 points correspond to the scene. `markers{camid}.png` shows the projection of 3d scene points (red) and labeled 2d points on the image (green). It indicates how well the extrinsic parameters are solved. 

9. `intrinsic_camera_params` contains two pickle files. You can also find the intrinsic parameters in `undistortion.py` file. 

### BamaPig3D_pure_pickle
This is a slim version of BamaPig3D, in which we remove `images`, `boxes_pr`, `keypoints_hrnet`, `masks_pr` folders. Only labeled images and labels are reserved. To save space, all label data are in `.pkl` format. `read_2dlabel_to_pickle` function in `visualize_BamaPig3D.py` shows how to encode 2D labels to pickle file. `label_mesh.pkl`, `label_3d.pkl` and `label_mix.pkl` are 70x4x19x3 matrices. `label_pose_params.pkl` is a dict seperating pose parameters to different parts, see information in the dict. 

## Demo code requirements
These functions are tested on Python 3.7 with conda virtual environment. The following python packages are necessary to run the codes in `code/` folder. Simply install the newest version. 
* scipy 
* numpy 
* opencv-python
* videoio 
* ipython 
* tqdm 
* matplotlib 
* pyflann

Specifically, after install anaconda (follow https://www.anaconda.com/ to install the newest version), you can create a conda virtual environment by running 
```shell
conda create -n MAMMAL python=3.7.9
conda activate MAMMAL
pip install scipy numpy opencv-python videoio 
pip install ipython 
pip install tqdm matplotlib pyflann-py3
```
It works for both windows 10 and ubuntu 20.04 (other mainstream windows and ubuntu version may work as well). If the installation of some packages fail, just try to install them again. If always fail, you may need to google the solution. 

## Demo code description
`utils.py` contains some keypoint structure definitions.

`visualize_BamaPig2D.py` tells how to load and visualize 2d labels onto images, and generate Supplementary Fig. 4b and 4d. 

`visualize_BamaPig3D.py` tells how to load 2d keypoints in BamaPig3D dataset and generate Supplementary Fig. 8c and 8d. 

`bodymodel_np.py` is used to drive the PIG model. You should have prepared model files of the PIG model before run this file. 

`assemble_BamaPig3D.py` shows the procedure of Supplementary Fig. 8b. 

`undistortion.py` contains the intrinsic calibration parameters

## Train SLEAP using BamaPig2D 
[SLEAP](https://sleap.ai/) and [DeepLabCut](https://github.innominds.com/DeepLabCut) are the most popular multiple-animal pose estimation methods. Here, we provide an instruction on how to use BamaPig2D dataset to train SLEAP. Note that, we train SLEAP instead of DeepLabCut because only SLEAP supports COCO style dataset currently. 

1. Install SLEAP v1.2.6 following their official instructions. 
2. Download `BamaPig2D_sleap.zip`, unzip it, and you will get `train_pig_cocostyle_sleap.json`, `eval_pig_cocostyle_sleap.json` and `full_pig_cocostyle_sleap.json` files. We recommend to train SLEAP using `train_pig_cocostyle_sleap.json` or `full_pig_cocostyle_sleap.json`. Let's take `train_pig_cocostyle_sleap.json` as example. Put it under `{BamaPig2D_path}/images/` folder first. We put it under `images` instead of `annotations` folder because SLEAP load images from the folder where `.json` file lives. 
3. After preparing the data, open SLEAP software, click `File->Import->COCO dataset`, wait for about half a minute before SLEAP load all the images. 
4. Click `Predict->Run training` to open the training setting panel. 
5. We recommend to use "top-down" structure. Set `Sigma for Centroids` as 8.00 and `Sigma for Nodes` as 4.00. Choose `unet` as model backbone, set `Max Stride` as 64. 
6. Click `Run` to start the training process. It may take half a day to finish the training. 

ATTENTION! Because BamaPig2D dataset contains 3003 images for training (3340 for full dataset), SLEAP requires at least 17.6GB GPU memory to train the top-down model. Therefore, I trained SLEAP using a single NVIDIA RTX 3090Ti GPU which has 24GB memory. If you could not access such a high-end GPU, you could remove some annotation samples from the json file. For example, for NVIDIA RTX 2080Ti (11GB), you may reduce the training images to 1000. 


## Citation
If you use these datasets in your research, please cite the paper

```BibTex
@article{MAMMAL, 
    author = {An, Liang and Ren, Jilong and Yu, Tao and Hai, Tang and Jia, Yichang and Liu, Yebin},
    title = {Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL},
    booktitle = {},
    month = {July},
    year = {2022}
}
```