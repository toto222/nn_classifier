# nn_classifier
A 3-layer neural network for image classification assignment

From the course “Neural Networks and Deep Learning” at the School of Data Science, Fudan University

Without the help of pytorch's automatic differentiation, all processes of the neural network are completed with only `numpy`, including forward propagation, gradient calculation and back propagation.

# Preparation
## Python3 is enough
```shell
pip install -r requirements.txt
```
## dataset
The data is already in this repo, to learn more you can go [`Fashion-MNIST`](https://github.com/zalandoresearch/fashion-mnist)

# Train & Evaluation
There are a number of options that can be set, most of which can be used by default, which you can view in `main_nn.py`.
## for train
```
python main_nn.py --act_func <acivation funcion> --learning_rate <base learning rate> \
                  --hidden <size of hidden layer> --epochs <epochs> --L2 <L2 regulation>
```
If you want to use a learning rate strategy, be sure to give the corresponding parameter together, for example
```
# Learning strategy for cosines of period 10
python main_nn.py <...> --schedule cos --sche_factor 10

# Exponential descent strategy with parameter 0.95
python main_nn.py <...> --schedule exp --sche_factor 0.95
```

## for evaluation
```
python main_nn.py --evaluate --filename <model_ckp_path>
```

