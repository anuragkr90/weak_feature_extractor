## Details "Knowledge Transfer From Weakly Labeled Audio Using Convolutional Neural Network For Sound Events And Scenes" Anurag Kumar, Maksim Khadkevich, Christian Fügen 
## ICASSP 2018

### https://arxiv.org/pdf/1711.01369.pdf

## Check out this webpage "http://www.cs.cmu.edu/~alnu/TLWeak.htm" for more results and details

### This code provides the  bare minimum to obtain audio representations using Deep CNN models trained on weakly labeled data (Audioset - Balanced set)

#### 1. call the main function in feat_extractor - returns 1024 or 527 dimensional features

#### 2. It will work with audio of any duration but I would suggest to pad it to make it at least 1.5 seconds for now. 

#### 3. You can turn on gpu use by 'usegpu' variable. Although, for very long audio (more than a few minutes) you might end up getting gpu memory error. 

#### The trained CNN is used to learn meaningful representations for a given audio recording. The classification task can be done by training another classifier on these representations (linear SVMs in this case). 

#### 4. class names and id in classes_id_name.txt for the 527 sounds in audioset over which the model was trained. 

#### 6. Doubts??? -  send me an email. 
