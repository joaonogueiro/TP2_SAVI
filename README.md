# TP2_SAVI

**Pratical Work 2  - SAVI - Where's my coffee mug?** <br>
SAVI - 2023/2024 Universidade de Aveiro
<br>
<br>
## Authors: 
### Group JJJ
- João Figueiredo (116189) vie.fig@ua.pt
- João Nogueiro (111807) joao.nogueiro@ua.pt
- José Nuno Cunha (95167) nunocunha99@ua.pt
 <br>

 ## Program Description
 This project was carried out as part of the "Advanced Industrial Vision Systems " subject, at the University of Aveiro. This system is able to process information that was taken from 3d sensors and also RGB cameras, the goal, is to use that information and detect some objects on random scenes and classify them by their properties, color, area, etc.
 It starts by pre-processing the 3D scenarios and isolates each object and compares it to the objects already in the database, then it classifies the object by it's matching class.


 ## Used data in the development of the program
 [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)
 [RGB-D Scenes Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)
 These are from: Washington RGB-D Dataset


 ## Libraries and dependencies installation
 These are the libraries that you have to have on your computer to run this program:
- `pip install opencv-python`
- `pip install open3d`
- `pip install gtts`
and some more
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
  
# Milestones

### Objetivo 1 - Treino de um classificador em deep learning

Pretende-se que o sistema desenvolvido seja capaz de detetar objetos na cena (em dados do RGB-D Scenes Dataset) e depois calcule várias propriedades dos objetos.

O RGB-D Object Dataset deve ser explorado e os seus dados utilizados para desenvolver uma rede de deep learning que consiga efetuar a classificação dos objetos.
Deve-se proceder à divisão do dataset em treino e teste (80% / 20%). Depois de treinada, deve ser possível calcular a precisão global e por classe.

### Objetivo 2 - Pre-processamento 3D

A ideia é desenvolver um sistema que processe a nuvem de pontos de uma cena (dados do RGB-D Scenes Dataset) e consiga isolar os objetos na nuvem de pontos. O sistema deve calcular várias propriedades dos objetos como:

- a cor,
- a altura,
- a largura,
- outras propriedades relevantes.

### Objetivo 3 - Classificação de objetos na cena

A segmentação de objetos nas nuvens de pontos que foi efetuada anteriormente pode ser utilizada para descobrir a zona onde está o objeto na imagem RGB. Daqui pode-se extrair uma sub-imagem que contenha apenas o objeto e dá-la à rede de classificação anteriormente desenvolvida.
Outra solução será treinar [uma rede de classificação que utilize informação tridimensional para o reconhecimento de objetos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf).

### Objetivo 4 - Descrição áudio da cena

A ideia é utilizar um sintetizador de discurso para que o sistema consiga descrever verbalmente a cena que lhe é mostrada, dizendo por exemplo: "A cena contém duas canecas, uma branca e outra azul, e uma caixa de cereais vermelha com altura de 15 centímetros."

### Objetivo 5 - Métricas de performance

Todos os algoritmos desenvolvidos deverão ser testados a apresentadas métricas de performance.
Em particular, para os detetores de objetos devem ser apresentadas métricas de [Precision, Recall e F1 Score](https://www.askpython.com/python/examples/precision-and-recall-in-python). Para problemas de classificação multi-class recomenda-se [este link](https://www.evidentlyai.com/classification-metrics/multi-class-metrics)

### Objetivo 6 - Sistema em tempo real

Utilizando uma câmara RGB-D em frente a uma mesa, experimente o sistema a correr em tempo real.

# Desenvolvimento

## treino de um classificador

## pré processamento
### ler a nuvem de pontos:
usar a funçao "o3d.io.read_point_cloud()" para ler a nuvem de pontos

diminuição da resolução "downsampling"
### remover o chão:

-> encontrar a mesa:
   