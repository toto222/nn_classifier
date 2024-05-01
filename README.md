# nn_classifier
This is a 3-layer neural network for image classification assignment.

From the course `"Neural Networks and Deep Learning"` at the `School of Data Science, Fudan University`

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

## for reference
In `ref_torch.py` is the same neural network implemented by pytorch, which you can use to compare with numpy's neural network if you're interested.

## for hyper parameters reseaech
`hyper_param_search.py` is used to search for appropriate hyperparameter settings, you can use it directly but it may consume more time.



