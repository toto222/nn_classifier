import os
import argparse
import numpy as np
from utils import mnist_reader
import NN
from NN import Dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser('Vision Models for Classification')
    # Training setting
    parser.add_argument('--batch_size', type=int, default=128, #64 128
                    help='batch_size')
    parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate')
    parser.add_argument('--L2', type=float, default=0.0001)
    parser.add_argument('--schedule', type=str, default=None,
                    choices=['cos', 'exp'],
                    help='choose lr strategy')
    parser.add_argument('--sche_factor', type=float,default=0. ,
                    help='factor of lr schedule')
    parser.add_argument('--act_func', type=str, default='sigmoid',
                    choices=['sigmoid', 'tanh','relu'],
                    help='choose activation function')
    
    # dataset & model
    parser.add_argument('--root', type=str, default='data/fashion',
                    help='dataset path')
    parser.add_argument('--split', type=float, default=0.8,
                    help='split train and val set')    
    parser.add_argument('--hidden', type=int, default=256)

    # other
    parser.add_argument('--seed', type=int, default=850011,
                    help='seed for initializing training')
    
    parser.add_argument('--model_dir', type=str, default='./save',
                    help='path to save models')
    parser.add_argument('--filename', type=str, default=None,
                    help='filename to load or save model')
    parser.add_argument('--evaluate', default=False,
                    action="store_true",
                    help='evaluate model test set')
    
    args = parser.parse_args()
    if not args.evaluate:
        args.filename = 'lr_{}_bsz_{}_act_{}_sche_{}_hidden-layer_{}'. \
            format(args.learning_rate, args.batch_size, 
                   args.act_func, args.schedule, args.hidden)
    return args

def cosine_lr(lr, epoch, T_max, lr_min=1e-6):
    lr_now = lr_min + (1 + np.cos(np.pi * epoch / T_max)) / 2 * (lr - lr_min)
    return lr_now
        
def exp_lr(lr,epoch, alpha=0.95):
    lr_now = lr * np.power(alpha, epoch)
    return lr_now

def train_one_epoch(model, train_data, optimizer):
    loss_sum = 0
    for batch in tqdm(train_data,total=len(train_data)):
        x,y = batch
        grads, loss = model.backward(x, y)
        optimizer.update(grads)
        loss_sum += loss
    print(f'train loss:{loss_sum}')
    return loss_sum
    
def validate(model, val_data):
    loss_sum = 0
    total = 0
    correct = 0
    for batch in tqdm(val_data,total=len(val_data)):
        x,y = batch
        y_pred = model(x)       
        m = y.shape[0]
        y_true = np.zeros((m, 10))
        y_true[np.arange(m), y] = 1
        loss = -1/m * np.sum(y_true * np.log(y_pred))
        loss_sum += loss
        
        predicted_labels = np.argmax(y_pred, axis=1)
        total += len(x)
        correct += (predicted_labels == y).sum()
    accuracy = 100 * correct / total   
    print(f'val loss:{loss_sum} accuracy:{accuracy}')
    return loss_sum, accuracy
       
    

def main():
    
    args = parse_option()
    
    np.random.seed(args.seed)
    data_path = args.root
    
    X_train, y_train = mnist_reader.load_mnist(data_path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')
    
    if args.evaluate:
        assert os.path.exists(args.filename)
        X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')
        model = NN.Model()
        model = model.load(args.filename)
    
        test_data = Dataloader([X_test/255, y_test])
        test_loss, test_acc = validate(model, test_data)
        
        return None
    
    nums = len(X_train)
    indices = np.arange(nums)
    np.random.shuffle(indices)
    
    train_ratio = args.split
      
    num_train = int(train_ratio * nums)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    
    X_train_set = X_train[train_indices]
    y_train_set = y_train[train_indices]
    X_val_set = X_train[val_indices]
    y_val_set = y_train[val_indices]
    
    batch_size = 128
    
    train_data = Dataloader([X_train_set/255, y_train_set],batch_size=batch_size)
    val_data = Dataloader([X_val_set/255, y_val_set],batch_size=batch_size)
    
    act = NN.__dict__[args.act_func]
    hidden = args.hidden
    
    model = NN.Model(hidden=hidden,act=act)
    
    base_lr = args.learning_rate
    optimizer = NN.SGD(model=model,lr=base_lr,l2=args.L2)
    
    epoch = args.epochs
    best_acc = 0
    best_model = NN.Model()
    acc_rec = []
    train_loss_rec = []
    val_loss_rec = []
    lr_rec = []
    
    for i in range(epoch):
        print(f'Training for Epoch {i+1}')
        lr = base_lr
        if args.schedule=='cos':
            lr = cosine_lr(lr,i,args.sche_factor)
        elif args.schedule=='exp':
            lr = exp_lr(lr,i,args.sche_factor)
            
        lr_rec.append(lr)
        optimizer.learning_rate = lr
        
        train_loss = train_one_epoch(model, train_data, optimizer)
        train_loss_rec.append(train_loss)
        
        val_loss, val_acc = validate(model, val_data)
        acc_rec.append(val_acc)
        val_loss_rec.append(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.clone()
    
        
    test_data = Dataloader([X_test/255, y_test],batch_size=batch_size)
    test_loss, test_acc = validate(best_model, test_data)
    
    os.makedirs(args.model_dir, exist_ok=True)
    best_model.save(os.path.join(args.model_dir,args.filename))
    
    # fig, ax1 = plt.subplots()
    # # 绘制损失曲线
    # color = 'tab:red'
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Train Loss', color=color)
    # ax1.plot(train_loss_rec, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    
    # ax2 = ax1.twinx()  
    
    # color = 'tab:blue'
    # ax2.set_ylabel('Validation Loss', color=color)  
    # ax2.plot(val_loss_rec, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    
    
    # plt.title('Training and Validation Loss')
    # plt.tight_layout()
    # plt.savefig('training.pdf',format='pdf')
    
    # plt.show()
    
    
    # plt.figure()
    # # 绘制精度曲线
    # plt.plot(acc_rec, color='green')
    # plt.title('Accuracy Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    
    # plt.tight_layout()
    # plt.savefig('acc.pdf',format='pdf')
    # plt.show()

    
if __name__ == "__main__":
    main()
    
