# A Dataset and Framework for Learning State-invariant Object Representations

## ObjectsWithStateChange (OWSC) Dataset 
We introduce a new dataset of 406 household objects from 21 categories, undergoing diverse state changes in addition to other transformations such as pose and viewpoint changes. 
The goal of introducing this dataset is to facilitate research in learning object representations that are invariant to state changes while also staying invariant to transformations induced by changes in viewpoint, pose, illumination, etc., for fine-grained recognition and retrieval. Details of the dataset are provided in our [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Sarkar_A_Dataset_and_Framework_for_Learning_State-invariant_Object_Representations_WACV_2026_paper.pdf).


<img width="640" height="480" alt="OWSC_thumbnail copy" src="https://github.com/user-attachments/assets/3b5a7c2f-e488-423e-96d9-0e50708452af" />


The OWSC dataset comprises 13837 images of 406 household objects from 21 categories, captured using smartphone cameras under various state changes from arbitrary viewpoints. 
The dataset is divided into two splits: 

### OWSC-SI for state invariance:

This split is to evaluate invariance across different states and other transformations. It comprises 11328 images of 331 objects, randomly partitioned into: 
- Test Set of 3,428 images (≈ 10 per object).
- Train Set of 7,900 images (≈ 24 per object) with no overlap between the two partitions. 

<img width="640" height="480" alt="OWSC-SI" src="https://github.com/user-attachments/assets/17ef86db-4711-4697-a2a8-378f8ad7ab7d" />

### OWSC-GN for generalization to novel objects:

This split is to test generalization to novel objects. It comprises 2509 images of 75 unseen objects (not present in OWSC-SI), randomly partitioned into: 
- Gallery Set: Images of one state per object
- Probe Set: Images from the remaining states per object, with no overlap between the two partitions. 

This split is to evaluate the recognition and retrieval of novel objects amidst state variations and other  transformations by correctly matching the probe images of the objects with their gallery images. 

<img width="640" height="480" alt="OWSCGN" src="https://github.com/user-attachments/assets/fa3c3f58-4d35-46b4-931f-81265307ea06" />



The datasets are organized such that the images of each object identity are stored in a separate subfolder with an integer ID indicating the object-identity. 
The mapping of object-identities to categories (also indicated by an integer ID) are also provided as a `split_o2c.npy` file. 

The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10YH6n6UfXCDRYJQbJ5lt8kciS4fuODyz?usp=sharing).


## Evaluation
### Requirements and Setup
Please clone this repo and install the dependencies using:
```bash
conda env create -f environment_owsc.yml
```
### Downloading datasets
Download the datasets (OWSC, ObjectPI [1], ModelNet-40 [2], and FG3D [3]) from [Google Drive](https://drive.google.com/file/d/12TE9yvSmP1oEclhWYLeU8fcKjiIVvPsd/view?usp=drive_link)

Please unzip the data.zip file using and place the datasets in a folder named data
```bash
unzip data.zip
```
We have organized these datasets such that the multi-view images of each object identity are stored in a separate subfolder with an integer ID indicating the object-identity. 
The train and test splits for the above-mentioned datasets can be downloaded from the link provided above.
The mapping of object-identities to categories is also provided as `train_o2c.npy` and `test_o2c.npy` files.

### Downloading our trained models
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1dpzTXwZZHAQH7b0H3ovJPbKtIiggBO-K?usp=drive_link)
and place them in a folder named model_weights 

## Benchmarking different methods using our dataset: 
In the paper, eight invariant recognition and retrieval tasks are proposed. These tasks are category and object-based where either a single or multiple images are used during inference. 
 
To evaluate different methods on these tasks, please run the following commands: 

### OWSC-SI split

For Our method (using curriculum learning): 
```bash
python evaluate_OWSC_SI.py ours model_weights/OWSC/PiRO2024/Ours_OWSCSI_CURRICULUM_nH1_nL2_2.pth 1 2
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/ecee0ade-4881-45e3-9ccc-c9957e4970c6" />

For PiRO method:
```bash
python evaluate_OWSC_SI.py piro model_weights/OWSC/PiRO2024/PiRO_OWSCSI_RAND_CATG_nH1_nL1.pth 1 1
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/9f1185e9-107b-469d-9265-8ba68e75f82b" />

### OWSC-GN split

For Our method (using curriculum learning): 
```bash
python evaluate_OWSC_GN.py ours model_weights/OWSC/PiRO2024/Ours_OWSCGN_CURRICULUM_nH1_nL2.pth 1 2
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/75853ffc-ffd9-4765-8c55-a70e453bd430" />


For PiRO method:
```bash
python evaluate_OWSC_GN.py piro model_weights/OWSC/PiRO2024/PiRO_OWSCGN_RAND_CATG_nH1_nL1.pth 1 1
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/686f23d8-c839-4bad-8d80-fedc9771462e" />

For PI-CNN, PI-Proxy, and PI-TC methods: 

```bash
python evaluate_OWSC_SI/GN.py picnn model_weights/OWSC/PIE2019/PICNN_1.0_1.0_1.0_1_150.pth 1 1
python evaluate_OWSC_SI/GN.py piprx model_weights/OWSC/PIE2019/PIPRX_1.0_1.0_1.0_1_150.pth 1 1
python evaluate_OWSC_SI/GN.py pitc model_weights/OWSC/PIE2019/PITC_1.0_0.2_1.0_1_150.pth 1 1
```
## Ablation for our Curriculum Learning Approach: 
For this ablation, we compare performance of the same dual-encoder model with same number of self-attention layer and heads trained using different object pair sampling strategies:
- Randomly sampling object pairs from the same category (models named as RAND_CATG)
- Mining object pairs based on our Curriculum Learning approach (models named as CURRICULUM)

For Random Sampling from Same Category using PiRO's dual-encoder architecture (with nHeads = 1, nLayers = 1)
```bash
python evaluate_curriculum.py OWSC model_weights/OWSC/PiRO2024/PiRO_OWSCSI_RAND_CATG_nH1_nL1.pth 1 1
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/9dc29e4e-2430-4e20-8a3b-725b3bbd5939" />

For Curriculum Learning using the same architecture (with nHeads = 1, nLayers = 1)
```bash
python evaluate_curriculum.py OWSC model_weights/OWSC/PiRO2024/PiRO_OWSCSI_CURRICULUM_nH1_nL1.pth 1 1
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/f92c6305-3df5-4d3c-8f2d-c99bcf542059" />


Similarly, for comparing performance with random sampling from same category (RAND_CATG) and Curriculum Learning (CURRICULUM) on pose-invariant tasks using the other multi-view datasets, please run the following commands: 

For ObjectPI (OOWL): 
```bash
python evaluate_curriculum.py OOWL model_weights/ObjectPI/RAND_CATG_OOWL_nH1_nL1.pth 1 1
python evaluate_curriculum.py OOWL model_weights/ObjectPI/CURRICULUM_OOWL_nH1_nL1.pth 1 1
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/2f3eee73-1d4a-4586-af60-32356445ca9f" />

For ModelNet-40: 
```bash
python evaluate_curriculum.py MNet40 model_weights/ModelNet40/RAND_CATG_MNet40_nH1_nL1.pth 1 1
python evaluate_curriculum.py MNet40 model_weights/ModelNet40/CURRICULUM_MNet40_nH1_nL1.pth 1 1 
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/9dd6ad93-a95d-458b-9c7f-3b721a2398e9" />


For FG3D: 
```bash
python evaluate_curriculum.py FG3D model_weights/FG3D/RAND_CATG_FG3D_nH1_nL1.pth 1 1
python evaluate_curriculum.py FG3D model_weights/FG3D/CURRICULUM_FG3D_nH1_nL1.pth 1 1 
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/f53989bf-f314-48a4-9be7-7cee54b151d2" />


## Ablation for Architecture: 
For this ablation, we compare performance of models with different number of self-attention layers trained using the same curriculum learning approach. 

For model with self-attention nHeads = 1 and nLayers = 1 trained using curriculum learning on our OWSC dataset: 
```bash
python evaluate_OWSC_SI.py ours model_weights/OWSC/PiRO2024/PiRO_OWSCSI_CURRICULUM_nH1_nL1.pth 1 1
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/f136211c-fac5-4f86-afee-cd76e1331f9a" />


For model with self-attention nHeads = 1 and nLayers = 2 trained using curriculum learning on our OWSC dataset: 
```bash
python evaluate_OWSC_SI.py ours model_weights/OWSC/PiRO2024/Ours_OWSCSI_CURRICULUM_nH1_nL2.pth 1 2
```
<img width="476" height="191" alt="image" src="https://github.com/user-attachments/assets/cc719751-1843-43e3-b2d4-39b01464f3d0" />



## References
[1] Chih-Hui Ho, Pedro Morgado, Amir Persekian, and Nuno Vasconcelos. PIEs: Pose invariant embeddings. In Computer Vision and Pattern Recognition (CVPR), 2019.  
[2] Zhirong Wu, S. Song, A. Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and J. Xiao. 3D shapenets: A deep representation for volumetric shapes. In Computer Vision and Pattern Recognition (CVPR), pages 1912–1920, Los Alamitos, CA, USA, 2015.  
[3] Xinhai Liu, Zhizhong Han, Yu-Shen Liu, and Matthias Zwicker. Fine-grained 3D shape classification with hierarchical part-view attentions. IEEE Transactions on Image Processing, 2021.  
[4] Rohan Sarkar, Avinash Kak; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 17077-17085

## Citation 
If you use the dataset or curriculum learning approach in your work, please cite our [paper](https://openaccess.thecvf.com/content/WACV2026/html/Sarkar_A_Dataset_and_Framework_for_Learning_State-invariant_Object_Representations_WACV_2026_paper.html): 
```
@InProceedings{Sarkar_2026_WACV,
    author    = {Sarkar, Rohan and Kak, Avinash},
    title     = {A Dataset and Framework for Learning State-invariant Object Representations},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {4715-4723}
}
```
