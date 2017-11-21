# CapsuleNetworks
A light-way system-wrapper of Capsule Layers using tensorpack
 
This code is mainly borrowed on mnist_convnet.py from @ppwwyyxx and CapsLayer.py from @naturomics 
 
Paper: Dynamic Routing Between Capsules by Sara Sabour, Nicholas Frosst, Geoffrey E Hinton 

https://arxiv.org/abs/1710.09829
 
<p align="center">
  <img src="/images/Architecture.PNG">
</p>

 
 ## Prerequisites
 ```bash
 sudo pip install tensorflow_gpu --upgrade # 1.4.0
 sudo pip install tensorpack --upgrade # 0.7.1
 ```
 
 ## Training
 ```python
 python mnist_capsnet_v2.py --gpu='0'
 ```
 
 ## Monitoring via tensorboard
 ```bash
 tensorboard --logdir='train_log'
 ```
<p align="center">
  <img src="/images/Visualization.PNG">
</p>

 
 
 
 
