# ObjectsWithStateChange (OWSC) Dataset 

## Datasets
We introduce a new dataset comprising 331 household objects from 21 categories undergoing diverse state changes in addition to other transformations such as pose and viewpoint changes. 
The goal of introducing this dataset is to facilitate research in learning object representations that are invariant to state changes while also staying invariant to transformations induced by changes in viewpoint, pose, illumination, etc. for fine-grained recognition and retrieval. 

<img width="1248" alt="Data_FG" src="https://github.com/user-attachments/assets/d6b56614-c68f-459e-956b-7a6301d15378">


For each object, two sets of images are collected: 
- StateChange dataset: Images of each object in diverse states are captured from multiple arbitrary viewpoints. Also, there are other transformations such as variations in pose, lighting conditions, and background. 
- Probe dataset: Images of each object placed in an unseen state are captured from arbitrary viewpoints.

![Screenshot 2024-07-18 at 12 47 37 PM](https://github.com/user-attachments/assets/40df2f97-67af-4e4e-b34c-74a6675ce990)


Overall, the StateChange dataset comprises 7900 images with an average of 24 images per object, and the Probe dataset comprises 3428 images
with approximately 10 images for each object. The StateChange dataset is used for training, while the Probe dataset comprising images of the same objects in an unseen state, is used for testing.

The datasets are organized such that the images of each object identity are stored in a separate subfolder with an integer ID indicating the object-identity. 
The mapping of object-identities to categories (also indicated by an integer ID) are also provided as a `obj2cat.npy` file. 
The text descriptions describing the visual characteristics of the object will be released soon. 

The StateChange and Probe datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/19icj12ccxArA7vpiuk-VT8fy5g-6S9Tu?usp=sharing).

If you use the dataset, please cite our [paper](https://arxiv.org/abs/2404.06470). 
