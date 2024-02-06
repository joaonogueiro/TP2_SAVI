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
 This project was carried out as part of the "Advanced Industrial Vision Systems " subject, at the University of Aveiro. This system is able to process information that was taken from 3d sensors and also RGB cameras, the goal, is to use that information and detect some objects on random scenes and classify them by their properties, color, area, etc.
 It starts by pre-processing the 3D scenarios and isolates each object, after that the program is able to create a boundingbox around each object and get their properties, color, area and volume, finally, the output of the program is the point clouds of the isolated objects of a certain scenario each one with a different color just to distinguish them in an easier way. Check the image below.
 
<p align="center">
  <img src="/home/nunocunha99/Desktop/MEAI/2ano/1sem/practical_works_savi/TP2_SAVI/2D_Classifier/Results/isolated_objects_preprocess3d.jpeg" alt="Alt text">
</p>


 After the 3D pre-processing, the system compares it to the objects already in the database by a , then it classifies the object by it's matching class.


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
<br>


## Demonstrational Video
Fazes um vídeo de demonstração rápida do programa.

 
## Evaluation Criteria

  | % | Objetivo  |    Tarefa     |
  | ------------- | ------------- |
  |15%| Objetivo 1|Treino de um classificador|
  |15%| Objetivo 2|Pré-processamento 3D|
  |15%| Objetivo 3|Classificação objetos na cena|
  |10%| Objetivo 4|Descrição audio|
  |15%| Objetivo 5|Métricas de performance|
  |5% | Objetivo 6|Sistema em tempo real|
  |10%| Objetivo 7|Aspeto geral do software|
  |5% | Objetivo 8|Extras|
  |10%| Objetivo 9|Código e Github|
  

   
