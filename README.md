# Short Project 2022/23 Q2 (VC)

## Dependències

- [Python3](https://www.python.org/downloads/)
- OpenCV
- Tensorflow v2

```bash
pip install opencv-python
pip install tensorflow
```



## Ús:

4 videos disponibles: 
- [`MotorcycleChase`](https://drive.google.com/file/d/1wnJzINqYj0qR6H0oFz2VlBjH2Rf3MmSH/view)
- [`PolarBear`](https://drive.google.com/file/d/1JoZSeyQVdbxbjFoOQ-hLMhSo_uvEFqC5/view)
- [`Elephants`](https://drive.google.com/file/d/1BzpNJh1Vbi20R73s1-a0G2kCx6I8Zl8i/view)
- [`Bike`](https://drive.google.com/file/d/1xQl8lj-U_uphvSq8Zl64q5RPqXQ77l5Z/view)

Per poder usar cada video (conjunt d'imatges), la carpeta ha d'estar en la mateixa carpeta que els scripts de manera que els frames estiguien en el path relatiu:
`MotorcycleChase/img/...`

### Tracking

#### Sift:

Canvia el video en la última línea del script

```bash
python3 sift.py
```

#### Histograma de color:

Canvia el video en la última línea del script

```bash
python3 histo.py
```

### Reconeixement

#### Viola Jones, Haar Cascade:

Descomenta la línea del video que vols executar

```bash
python3 sift.py
```

#### Single Shot Detection (SSD):

Usa l'argument `video` per seleccionar un dels quatre disponibles

```bash
python3 reconeixementSSD.py --video="Bike"
```
