# deeply-recursive-cnn-tf

## overview
This project is a test implementation of ["Deeply-Recursive Convolutional Network for Image Super-Resolution", CVPR2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.pdf) using tensorflow


Paper: ["Deeply-Recursive Convolutional Network for Image Super-Resolution"] (https://arxiv.org/abs/1511.04491) by Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea


Training highly deep CNN layers is so hard. However this paper makes it with some tricks like sharing filter weights and using intermediate outputs to suppress divergence in training. The model in the papaer contains 20 CNN layers without no any max-pooling layers, I feel it's amazing.

Also the paper’s super-resolution results are so nice. :)


## model structure

Those figures are from the paper. There are 3 different networks which cooperates to make images fine.
![alt tag](https://github.com/jiny2001/deeply-recursive-cnn-tf/documents/figure1.png)
![alt tag](https://github.com/jiny2001/deeply-recursive-cnn-tf/documents/figure3.png)

This model below is made by my code and drawn by tensorboard.
![alt tag](https://github.com/jiny2001/deeply-recursive-cnn-tf/documents/model.png)


## requirements

tensorflow, scipy, numpy and pillow


## how to use

```
# training with default parameters and evaluate with Set5 (takes whole day to train with high-GPU)
python main.py

# training with simple model (will be nice for CPU computation)
python main.py —initial_lr 0.001 —end_lr 1e-4 —feature_num 32 —inference_depth 5

# evaluation for set14 only (after training has done)
# [set5, set14, bsd100, urban100, all] are available
python main.py —dataset set14

# train for x4 scale images
# (you need to download image_SRF_4 dataset for Urban100 and BSD100)
python main.py —scale 4
```

Network graphs and weights / loss summaries are saved in **tf_log** directory.

Weights are saved in **model** directory.


## sample result

will be uploaded soon. So far I got a less PSNR compared to their paper yet there are some interesting results.

## datasets

Some images are already set in **data** folder yet some x3 and x4 images are not contained because of their data sizes.

for training:
+ ScSR [[ Yang et al. TIP 2010 ]](http://www.ifp.illinois.edu/%7Ejyang29/ScSR.htm)
( J. Yang, J. Wright, T. S. Huang, and Y. Ma. Image super- resolution via sparse representation. TIP, 2010 )

for evaluation:
+ Set5 [[ Bevilacqua et al. BMVC 2012 ]] (http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
+ Set14 [[ Zeyde et al. LNCS 2010 ]] (https://sites.google.com/site/romanzeyde/research-interests)
+ BSD100 [[ Huang et al. CVPR 2015 ]] (https://sites.google.com/site/jbhuang0604/publications/struct_sr) (only x2 images in this project)
+ Urban100 [[ Martin et al. ICCV 2001 ]] (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) (only x2 images in this project)


## disclaimer

Some details are not shown in the paper and my guesses are not enough, so some parts, especially for initial values, may be not good in some cases.

## acknowledgments

Thanks a lot for Assoc. Prof. **Masayuki Tanaka** at Tokyo Institute of Technology and **Shigesumi Kuwashima** at Viewplus inc.
