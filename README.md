# TP2_SAVI
Trabalho elaborado no âmbito da unidade curricular de Sistemas Avançados de Visão Industrial da Universidade de Aveiro.


UPDATE:
 - ficheiro class_teaching.py lê as imagens da pasta "teach_dataset", obtém o nome da classe de cada imagem e guarda as classes num array.
 - ficheiro obect_classifier.py lê as imagens que tiver que ler, faz um retângulo à volta do objeto de modo a identificar o objeto, após isto
 é aplicar feature matching e comparar a imagem atual com a imagem modelo da class e verificar a accuracy/confidence, se for acima de 0,85 (suponhamos), pertence à classe, se for abaixo, continua a iterar pelas imagens modelo das classes. 

 a pasta "teach_dataset" deve estar dentro da pasta "rgbd-dataset" sendo que esta está no mesmo local do script a executar, isto eventualmente pode alterar, mas para já, está assim.
 
 # Tópicos e percentagens para avaliação

  | Percentagem  | Objetivo |Tarefa|
| ------------- | ------------- |---------|
  |15%| Objetivo 1|Treino de um classificador|
  |15%| Objetivo 2|Pré-processamento 3D|
  |15%| Objetivo 3|Classificação objetos na cena|
  |10%| Objetivo 4|Descrição audio|
  |15%| Objetivo 5|Métricas de performance|
  |5% | Objetivo 6|Sistema em tempo real|
  |10%| Objetivo 7|Aspeto geral do software|
  |5% | Objetivo 8|Extras|
  |10%| Objetivo 9|Código e Github|
  
# Objetivos

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
   