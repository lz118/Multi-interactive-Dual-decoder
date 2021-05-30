# Multi-interactive-Dual-decoder-for-RGBT-Salient-Object-Detection

The pytorch implementation of Multi-interactive Siamese Decoder for RGBT Salient Object Detection
https://arxiv.org/pdf/2005.02315v2.pdf

![framework](./fig/framework.png)

## Train

- We use VT5000-Train to train our network. All the datasets are available in https://github.com/lz118/RGBT-Salient-Object-Detection
- The pretrained  model (VGG16) can be downloaded at https://pan.baidu.com/s/11lq3mUGRFP7TFvH9Eui14A [3513]


## Test

- The trained models on RGB-T Dataset 

  https://pan.baidu.com/s/1Wj6bfi7lhp1KF5iCSVj0gQ [4zkx]
  
  https://drive.google.com/file/d/11lU5TaRZMTXQ6QCbBLinG9iDvIUDrRP5/view?usp=sharing

- The trained models on RGB-D Dataset 

  https://pan.baidu.com/s/1KlAKrVszQisG0bK1kiedzA [2ulc]
  
  https://drive.google.com/file/d/1LKVn3iPDBI07DUBiirm4bk2-7yA3pTSM/view?usp=sharing

## Evalution

- For RGB-T SOD, we provide the our saliency maps on VT821, VT1000 and VT5000-Test. 

  https://pan.baidu.com/s/1hEZJyEJ2j1n1JKUgUaZZLQ  [0div]


- For RGB-D SOD, we provide the our saliency maps on SIP, SSD，STERE，LFSD and DES. 

  https://pan.baidu.com/s/1ZHxvMh818RxlZGW1hQA70w  [2oqx]

- The evalution toolbox is provided by https://github.com/jiwei0921/Saliency-Evaluation-Toolbox
