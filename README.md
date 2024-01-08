# TP2_SAVI
Trabalho elaborado no âmbito da unidade curricular de Sistemas Avançados de Visão Industrial da Universidade de Aveiro.


UPDATE:
 - ficheiro class_teaching.py lê as imagens da pasta "teach_dataset", obtém o nome da classe de cada imagem e guarda as classes num array.
 - ficheiro obect_classifier.py lê as imagens que tiver que ler, faz um retângulo à volta do objeto de modo a identificar o objeto, após isto
 é aplicar feature matching e comparar a imagem atual com a imagem modelo da class e verificar a accuracy/confidence, se for acima de 0,85 (suponhamos), pertence à classe, se for abaixo, continua a iterar pelas imagens modelo das classes. 

 a pasta "teach_dataset" deve estar dentro da pasta "rgbd-dataset" sendo que esta está no mesmo local do script a executar, isto eventualmente pode alterar, mas para já, está assim.
