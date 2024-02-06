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
 It starts by pre-processing the 3D scenarios and isolates each object, after that the program is able to create a boundingbox around each object and get their properties, color, area and volume, finally, the output of the program is the point clouds of the isolated objects of a certain scenario each one with a different color just to distinguish them in an easier way as you can see by the image below.
 
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
<br>

## User's Guide
Describe how the use can make the system work in his computer

To run this system you have to follow these steps:

1st. Clone this project's repository:
```bash
git clone https://github.com/joaonogueiro/TP2_SAVI.git
```
After cloning the repository, run the 3D pre-processing script to get the object's point clouds. 
For that, you have to change the directory to:
```bash
cd TP2_SAVI/pre_proce_3d/auto_found_table/final_main.py
```

Then, after getting the point clouds of certain objects, go to:
```bash
cd TP2_SAVI/2D_Classifier/
```
And finally run the "main.py" file.

<br>


## Project Results
 IMAGEM 1
 IMAGEM 2
 IMAGEM 3
 IMAGEM 4


 
## Project Evaluation Criteria

  | Percentagem | Objetivo | Tarefa |
| :---:         |     :---:      |
| 15%   | Objetivo 1     |   Treino de um classificador       |     
| 15%     | Objetivo 2       |   Pré-processamento 3D       |       
| 15%     | Objetivo 3       |   Classificação objetos na cena       |
| 10%    | Objetivo 4       |   Descrição audio       |
| 15%      | Objetivo 5       |   Métricas de performance       | 
| 5%      | Objetivo 6       |   Sistema em tempo real       |
| 10%      | Objetivo 7       |   Aspeto geral do software       |
| 5%      | Objetivo 8       |   Extras       |
| 10%      | Objetivo 9       |   Código e Github       |  

   
