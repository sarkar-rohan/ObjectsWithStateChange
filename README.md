# A Dataset and Framework for Learning State-invariant Object Representations 
(WACV 2026)

## ObjectsWithStateChange (OWSC) Dataset 
We introduce a new dataset of 406 household objects from 21 categories, undergoing diverse state changes in addition to other transformations such as pose and viewpoint changes. 
The goal of introducing this dataset is to facilitate research in learning object representations that are invariant to state changes while also staying invariant to transformations induced by changes in viewpoint, pose, illumination, etc., for fine-grained recognition and retrieval. 

<img width="950" height="540" alt="OWSC_thumbnail copy" src="https://github.com/user-attachments/assets/3b5a7c2f-e488-423e-96d9-0e50708452af" />


The OWSC dataset comprises 13837 images of 406 household objects from 21 categories, captured using smartphone cameras under various state changes from arbitrary viewpoints. 
The dataset is divided into two splits: 

### OWSC-SI for state invariance:

This split is to evaluate invariance across different states and other transformations. It comprises 11328 images of 331 objects, randomly partitioned into: 
- Test Set of 3,428 images (≈ 10 per object).
- Train Set of 7,900 images (≈ 24 per object) with no overlap between the two partitions. 

<img width="960" height="540" alt="OWSC-SI" src="https://github.com/user-attachments/assets/17ef86db-4711-4697-a2a8-378f8ad7ab7d" />

### OWSC-GN for generalization to novel objects:

This split is to test generalization to novel objects. It comprises 2509 images of 75 unseen objects (not present in OWSC-SI), randomly partitioned into: 
- Gallery Set: Images of one state per object
- Probe Set: Images from the remaining states per object, with no overlap between the two partitions. 

This split is to evaluate the recognition and retrieval of novel objects amidst state variations and other  transformations by correctly matching the probe images of the objects with their gallery images. 

<img width="960" height="540" alt="OWSCGN" src="https://github.com/user-attachments/assets/fa3c3f58-4d35-46b4-931f-81265307ea06" />



The datasets are organized such that the images of each object identity are stored in a separate subfolder with an integer ID indicating the object-identity. 
The mapping of object-identities to categories (also indicated by an integer ID) are also provided as a `split_o2c.npy` file. 

The text descriptions of the visual characteristics of the objects will be released soon. 

The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/19icj12ccxArA7vpiuk-VT8fy5g-6S9Tu?usp=sharing).

## Evaluation
### Requirements and Setup
Please clone this repo and install the dependencies using:
```bash
conda env create -f environment_owsc.yml
```
### Downloading datasets
Download the datasets (OWSC, ObjectPI, ModelNet-40, and FG3D) from [Google Drive](https://drive.google.com/file/d/1r_WKcmkemumC79VglA8LVgUTZ9J69d9L/view?usp=drive_link)

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

For Our method (using curriculum learning): 
```bash
python evaluate_OWSC.py ours model_weights/OWSC/PiRO2024/Ours_PiRO_CURRICULUM_nH1_nL2_2.pth 1 2
```

<img width="652" alt="Ours_result" src="https://github.com/user-attachments/assets/ecd1b61a-f5d3-412a-8cf3-69e32643d37a" />


For PiRO method:
```bash
python evaluate_OWSC.py piro model_weights/OWSC/PiRO2024/PiRO_RAND_CATG_nH1_nL1.pth 1 1
```
<img width="652" alt="ification Accuracy Category 87 07701283547257 $" src="https://github.com/user-attachments/assets/0f1ab731-e4c1-4bbd-b10f-f94fc34debe0" />

For PI-CNN, PI-Proxy, and PI-TC methods: 

```bash
python evaluate_OWSC.py picnn model_weights/OWSC/PIE2019/PICNN_1.0_1.0_1.0_1_150.pth 1 1
python evaluate_OWSC.py piprx model_weights/OWSC/PIE2019/PIPRX_1.0_1.0_1.0_1_150.pth 1 1
python evaluate_OWSC.py pitc model_weights/OWSC/PIE2019/PITC_1.0_0.2_1.0_1_150.pth 1 1
```
## Ablation for our Curriculum Learning Approach: 
For this ablation, we compare performance of the same dual-encoder model with same number of self-attention layer and heads trained using different object pair sampling strategies:
- Randomly sampling object pairs from the same category (models named as RAND_CATG)
- Mining object pairs based on our Curriculum Learning approach (models named as CURRICULUM)

For Random Sampling from Same Category using PiRO's dual-encoder architecture (with nHeads = 1, nLayers = 1)
```bash
python evaluate_curriculum.py OWSC model_weights/OWSC/PiRO2024/PiRO_RAND_CATG_nH1_nL1.pth 1 1
```
<img width="652" alt="ification Accuracy Category 87 07701283547257 $" src="https://github.com/user-attachments/assets/fc185e73-8b4e-404a-9366-d70031cd60b6" />

For Curriculum Learning using the same architecture (with nHeads = 1, nLayers = 1)
```bash
python evaluate_curriculum.py OWSC model_weights/OWSC/PiRO2024/PiRO_CURRICULUM_nH1_nL1.pth 1 1
```
<img width="649" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/5f5efd1f-3d5e-4153-b439-3b1b41dd5923" />

Similarly, for comparing performance with random sampling from same category (RAND_CATG) and Curriculum Learning (CURRICULUM) on pose-invariant tasks using the other multi-view datasets, please run the following commands: 

For ObjectPI (OOWL): 
```bash
python evaluate_curriculum.py OOWL model_weights/ObjectPI/RAND_CATG_OOWL_nH1_nL1.pth 1 1
python evaluate_curriculum.py OOWL model_weights/ObjectPI/CURRICULUM_OOWL_nH1_nL1.pth 1 1 
```
For ModelNet-40: 
```bash
python evaluate_curriculum.py MNet40 model_weights/ModelNet40/RAND_CATG_MNet40_nH1_nL1.pth 1 1
python evaluate_curriculum.py MNet40 model_weights/ModelNet40/CURRICULUM_MNet40_nH1_nL1.pth 1 1 
```
For FG3D: 
```bash
python evaluate_curriculum.py FG3D model_weights/FG3D/RAND_CATG_FG3D_nH1_nL1.pth 1 1
python evaluate_curriculum.py FG3D model_weights/FG3D/CURRICULUM_FG3D_nH1_nL1.pth 1 1 
```

## Ablation for Architecture: 
For this ablation, we compare performance of models with different number of self-attention layers trained using the same curriculum learning approach. 

For model with self-attention nHeads = 1 and nLayers = 1 trained using curriculum learning on our OWSC dataset: 
```bash
python evaluate_OWSC.py ours model_weights/OWSC/PiRO2024/PiRO_CURRICULUM_nH1_nL1.pth 1 1
```
<img width="649" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/89859ae1-e180-4b72-ba7a-db327970ce62" />


For model with self-attention nHeads = 1 and nLayers = 2 trained using curriculum learning on our OWSC dataset: 
```bash
python evaluate_OWSC.py ours model_weights/OWSC/PiRO2024/Ours_PIRO_CURRICULUM_nH1_nL2.pth 1 2
```
<img width="653" alt="Ification Accuracy Category 89 206" src="https://github.com/user-attachments/assets/d6045967-e006-4ace-8f30-cf7f2aeb387b" />


## Citation 
If you use the dataset or curriculum learning approach in your work, please cite our [paper](https://arxiv.org/abs/2404.06470): 
```
@misc{sarkar2025datasetframeworklearningstateinvariant,
      title={A Dataset and Framework for Learning State-invariant Object Representations}, 
      author={Rohan Sarkar and Avinash Kak},
      year={2025},
      eprint={2404.06470},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.06470}, 
}
```
