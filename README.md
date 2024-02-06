# TP2_SAVI

**Pratical Work 2  - SAVI - Where's my coffee mug?** <br>
SAVI - 2023/2024 Universidade de Aveiro
<br>
<br>
## Authors: 
### Group 5
- João Figueiredo (116189) vie.fig@ua.pt
- João Nogueiro (111807) joao.nogueiro@ua.pt
- José Nuno Cunha (95167) nunocunha99@ua.pt
 <br>

 ## Program Description
 This project was carried out as part of the "Advanced Industrial Vision Systems " subject, at the University of Aveiro, it was developed in OpenCV using Python. 
 This system is able to process information that was taken from 3d sensors and also RGB cameras, the goal, is to use that information and detect some objects on random scenes and classify them by their properties, color, area, etc.
 It starts by pre-processing the 3D scenarios and isolates each object, after that the program is able to create a boundingbox around each object and get their properties, color, area and volume, finally, the output of the program is the point clouds of the isolated objects of a certain scenario each one with a different color just to distinguish them in an easier way as you can see by the image below, also, the program is able to tell you the properties of each object like mencioned above.
 
<p align="center">
  <img src="/2D_Classifier/Results/isolated_objects_preprocess3d.jpeg" alt="Alt text">
</p>

 After the 3D pre-processing, the system compares the object's point cloud got by the 3D pre-processing, with the objects already in the database, then it classifies the object through the object's point cloud coordinates and then gets it's matching class through.


 ## Used data in the development of the program
 [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)
 [RGB-D Scenes Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)
 These are from: Washington RGB-D Dataset


 ## Libraries and dependencies to install
 These are the libraries that you have to have on your computer to run this program:
- `pip install opencv-python`
- `pip install open3d`
- `pip install gtts`
- `pip install pcl_viewer`
- `sudo apt install libespeak-dev`
<br>

## User's Guide
Describe how the use can make the system work in his computer

To run this system you have to follow these steps:

1st. Clone this project's repository:
```bash
git clone https://github.com/joaonogueiro/TP2_SAVI.git
```
2nd. After cloning the repository, run the 3D pre-processing script to get the object's point clouds. 
For that, you have to change the directory to:
```bash
cd TP2_SAVI/pre_proce_3d/auto_found_table/main_3dpreprocess.py
```

3rd. Then, after getting the point clouds of certain objects, go to:
```bash
cd TP2_SAVI/2D_Classifier/
```
And finally run the "main.py" file.

<br>


## Project Results
<summary><b>Training/Validation Loss</b></summary>

<p align="center">
  <img src="/2D_Classifier/Results/Traning_Validation Loss.png" alt="Alt text">
</p>
 
<summary><b>Confusion Matrix</b></summary>

<p align="center">
  <img src="/2D_Classifier/Results/Confusion Matrix.png" alt="Alt text">
</p>

<summary><b>Class Metrics</b></summary>

<p align="center">
  <img src="/2D_Classifier/Results/Metrics.png" alt="Alt text">
</p>

<summary><b>3D Pre-Processing</b></summary>
This is the output of the 3D Pre-Processing process, also, in the terminal is shown the properties of each object, so that the
user can evaluate this whole process.
<p align="center">
  <img src="/2D_Classifier/Results/isolated_objects_preprocess3d.jpeg" alt="Alt text">
</p>



 
## Project Evaluation Criteria

| % | Objetivo  |    Tarefa     |
| :---: | :---: | :---:
|15%| Objetivo 1|Treino de um classificador|
|15%| Objetivo 2|Pré-processamento 3D|
|15%| Objetivo 3|Classificação objetos na cena|
|10%| Objetivo 4|Descrição audio|
|15%| Objetivo 5|Métricas de performance|
|5% | Objetivo 6|Sistema em tempo real|
|10%| Objetivo 7|Aspeto geral do software|
|5% | Objetivo 8|Extras|
|10%| Objetivo 9|Código e Github|

   
