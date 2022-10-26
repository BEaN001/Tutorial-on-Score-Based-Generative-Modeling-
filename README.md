# Tutorial-on-Score-Based-Generative-Modeling-
score based generative model from yang song
https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=0H1Rq5DTmW8o

ODE
速度最快 质量最差
Euler
质量还可以 速度比ODE慢
PC
质量最好 速度最慢

# Score-Based Generative Modeling


### Goals
This is a hitchhiker's guide to score-based generative models, a family of approaches based on [estimating gradients of the data distribution](https://arxiv.org/abs/1907.05600). They have obtained high-quality samples comparable to GANs (like below, figure from [this paper](https://arxiv.org/abs/2006.09011)) without requiring adversarial training, and are considered by some to be [the new contender to GANs](https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/).

![ncsnv2](https://github.com/ermongroup/ncsnv2/blob/master/assets/samples.jpg?raw=true)

The contents of this notebook are mainly based on the following paper: 

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "[Score-Based Generative Modeling through Stochastic Differential Equations.](https://arxiv.org/pdf/2011.13456.pdf)" Internation Conference on Learning Representations, 2021
