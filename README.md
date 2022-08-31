# Deep Q-Networks en juegos de Atari
Scritps para entrenar y testear modelos que jueguen a los juegos Breakout, Space Invaders y Boxing de Atari 2600 utilizando Deep Q-Networks. Se incluyen además modelos preentrenados con distintos hiperparámetros, que fueron utilizados para analizar cómo afectan el entrenamiento y la recompensa obtenida por el agente en cada uno de los juegos, que es parte del trabajo realizado en mi trabajo final para la Licenciatura en Ciencias de la Computación.

# Setup
Todas las herramientas son compatibles con Python 3.7, por lo que se recomienda esta versión.

1. [Instalar CUDA](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781)
2. Instalar [PyTorch](https://pytorch.org) con soporte para GPU
```powershell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
3. Instalar [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
```
pip install stable-baselines3[extra]
```
4. Instalar [OpenAI Gym](https://github.com/openai/gym)
```powershell
pip install gym
```
5. Obtener las ROMs de los juegos de Atari y extraer los archivos en un directorio
6. Importar las ROMs a OpenAI Gym
```powershell
python -m atari_py.import_roms [directorio de las ROMs]
```

# Uso
## Probar modelos
En el directorio correspondiente al juego que se quiera ejecutar (Breakout, Space Invaders o Boxing), introducir el siguiente comando en la terminal
```powershell
python playDQN.py
```
Luego, seleccionar el modelo a utilizar. 
El modelo jugará 100 partidas y se mostará como resultado la recompensa media obtenida en esa cantidad de partidas.

## Entrenar modelos
En el directorio correspondiente al juego para el que se quiera entrenar el modelo (Breakout, Space Invaders o Boxing), ejecutar
```powershell
python trainDQN.py
```
Además, se puede monitorear el entrenamiento utilizando TensorBoard
```powershell
tensorboard --logdir [directorio a utilizar para el monitoreo]
```
El modelo será guardado en la carpeta Modelos del directorio y su nombre será el nombre del juego seguido de DQN.
Notar que el modelo será entrenado con los hiperparámetros mostrados en [el paper original de Deep Q-Networks](https://arxiv.org/abs/1312.5602) presentado por DeepMind en 2013 y se hará en 10.000.000 de *timesteps*, que es número total de pasos que el agente hará en el ambiente. Se llamará a estos modelos "modelos base".

# Resultados
Las recompensas obtenidas por los distintos modelos en cada uno de los juegos se muestran en la siguiente tabla
| Modelo                              | Breakout       | Space Invaders | Boxing       |
|-------------------------------------|----------------|----------------|--------------|
|     Modelos Base                    |     287,02     |     736,5      |     92,4     |
|     Framestack = 2                  |     328,8      |     880,1      |     97,94    |
|     Framestack   = 6                |     328,24    |     795,45     |     86,55    |
|     Learning Rate = 0,001           |     2          |     285        |     -25      |
|     Learning   Rate = 0,00001       |     54,88      |     1057,3     |     97,48    |
|     Batch Size = 16                 |     262,63     |     823,75     |     86,83    |
|     Batch   Size = 64               |     221,44     |     730,1      |     94,81    |
|     Exploration Fraction = 0,05     |     317,85     |     705,25     |     90,9     |
|     Exploration   Fraction = 0,2    |     255,52     |     791,7      |     92,8     |
|     Buffer Size = 50.000            |     274,15*    |     950,05     |     93,35    |
|     Buffer   Size = 200.000         |     287,28     |     904,95     |     91,11    |

Como se puede observar, el agente que obtuvo una mayor recompensa en Breakout fue aquel que utiliza un framestack de 2 imágenes, con una media de 328,8 puntos. En Space Invaders fue el agente que utiliza un learning rate de 0,00001, con una media de 1057,3 puntos. En Boxing, nuevamente el agente con un framestack de 2 imágenes vuelve a ser aquel que obtiene el mayor puntaje medio, con 97,94 puntos.
