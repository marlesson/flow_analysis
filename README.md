# Flow Analysis

## Etapas

* Cálculo de Perspectiva (Ajuste de câmera)
* Optical Flow e cálculo dos vetores
* Agrupamento e Fluxo
* Cálculo de vetor por grupo 

# Uso

Cria cópias do Video usando a Piramide Gaussiana para análise multiescala

```
python pyramid_video.py -f data/scene01_x1_view1_1.avi -out multscale/scene01_x4_view1 -max_layer 3
```

Processa o vídeo e cria os vetores

```
python parse_lk.py -f data/sample.flv -s 3 -c 300
```

Processa os vetores e cria representação por blocos
```
python parse_arrows.py -f experimentos/output/scene06_x1.csv -out experimentos/output
```

## Métricas

* Fluxo
* Densidade
* Velocidade ?


## Optical Flow 

* https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
* https://vision.in.tum.de/research/optical_flow_estimation
* http://www.cs.cmu.edu/~saada/Projects/CrowdSegmentation/
* http://doras.dcu.ie/21880/1/PID4904817_Camera_Ready.pdf
* https://www.eurasip.org/Proceedings/Eusipco/Eusipco2010/Contents/papers/1569292551.pdf
* https://ieeexplore.ieee.org/document/7096587/
* https://www.coursera.org/learn/deep-learning-in-computer-vision/lecture/2wMqQ/optical-flow
* https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/

## Lucas-Kanade

* http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf
* https://pdfs.semanticscholar.org/6841/d3368d4dfb52548cd0ed5fef29199d14c014.pdf
* https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/
* https://www.youtube.com/watch?v=1r8E9uAcn4E
* https://www.youtube.com/watch?v=TcnVeK6YjUc


## Dataset

* https://www.sites.univ-rennes2.fr/costel/corpetti/agoraset/Site/AGORASET.html
* http://www.cs.cmu.edu/~saada/Projects/CrowdSegmentation/
* http://www.cvg.reading.ac.uk/PETS2009/a.html