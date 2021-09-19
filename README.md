# D3A
Data and demo code of our submission "Description and Demonstration Guided Data Augmentation for Sequence Tagging" to WWWJ.

##1. Introduction
* In **D3A-DATA**, we present four sequence tagging datasets used in our paper (Restaurant, Laptop, WNUT16, WNUT17.)
* In **D3A-DEMO-GLOVE-ABSA**, we present the demo code of D3A for ABSA with the GloVe-based sequence tagger.

##2. Requirements
* python 3.6.7
* pytorch 1.5.0
* pytorch-pretrained-bert 0.4.0
* numpy 1.19.1

##3. Usage
Go into **D3A-DEMO-GLOVE-ABSA**, and execute **bash DEMO.sh**.

After running with five seeds, the improvement brought by D3A over the baseline NoAug is about 11.94%.

