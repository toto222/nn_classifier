import dill
import numpy as np


def relu(x):
    # x = np.array(x)
    return np.maximum(0,x)

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -1/m * np.sum(y_true * log(y_pred))
    
    return loss

def log(x):
    return np.log(x+1e-9)

class Model:
    def __init__(self, size =784, hidden=256, act=sigmoid):
        self.fc1 = np.random.randn(size, hidden)*0.1
        self.bias1 = np.random.randn(1,hidden)*0.
        self.fc2 = np.random.randn(hidden,hidden)*0.1
        self.bias2 = np.random.randn(1,hidden)*0.
        self.fc3 = np.random.randn(hidden,10)*0.1
        self.bias3 = np.random.randn(1,10)*0.
        self.act = act
        self.params = {'fc1': self.fc1, 'bias1': self.bias1, 
                       'fc2': self.fc2, 'bias2': self.bias2,'fc3':self.fc3, 'bias3':self.bias3}
        self.grads = None

    def clone(self):
        
        clone_model = Model()

        clone_model.fc1 = np.copy(self.fc1)
        clone_model.bias1 = np.copy(self.bias1)
        clone_model.fc2 = np.copy(self.fc2)
        clone_model.bias2 = np.copy(self.bias2)
        clone_model.fc3 = np.copy(self.fc3)
        clone_model.bias3 = np.copy(self.bias3)
        clone_model.act = self.act

        clone_model.params = {'fc1': clone_model.fc1, 'bias1': clone_model.bias1,
                              'fc2': clone_model.fc2, 'bias2': clone_model.bias2,
                              'fc3': clone_model.fc3, 'bias3': clone_model.bias3}
        return clone_model

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)


    def load(self, filename):
        with open(filename, 'rb') as f:
            model = dill.load(f)
        return model
 
    
    def df(self,x):
        df = (self.act(x+1e-6)-self.act(x-1e-6))/2e-6
        return df
            
    def softmax(self,x):
        if x.ndim ==1: x = x.reshape(1,-1)
        # x -= x.max(axis=1, keepdims=True)
        x = np.exp(x)
        return x/x.sum(axis=1, keepdims=True)
    
    def forward(self,x):
        if x.ndim ==1: x = x.reshape(1,-1)
        x = self.act(x @ self.fc1 + self.bias1)
        x = self.act(x @ self.fc2 + self.bias2)
        x = x @ self.fc3 + self.bias3
        return self.softmax(x)
    
    def __call__(self,x):
        if x.ndim ==1: x = x.reshape(1,-1)

        return self.forward(x)

    def backward(self, x, y):
        if x.ndim ==1: x = x.reshape(1,-1)
        m = x.shape[0]
        
        y_true = np.zeros((m, 10))
        y_true[np.arange(m), y] = 1
        
        # 前向传播
        x1 = x @ self.fc1 + self.bias1
        x2 = self.act(x1)
        y1 = x2 @ self.fc2 + self.bias2
        y2 = self.act(y1)
        z1 = y2 @ self.fc3 + self.bias3
        # import pdb;pdb.set_trace()
        out = self.softmax(z1)
        
        loss = cross_entropy_loss(y_true,out)
        
        assert not np.isnan(loss).any(), "Value error"
        # if loss is np.nan:
        #     import pdb;pdb.set_trace()

        # 反向传播
               
        dw3 = y2.T @ (out-y_true)
        db3 = (out-y_true)
        db3 = db3.sum(axis=0,keepdims=True)
        # import pdb;pdb.set_trace()
        dw2 = x2.T @ ((out-y_true) @ self.fc3.T * self.df(y1))
        db2 = (out-y_true) @ self.fc3.T * self.df(y1)
        db2 = db2.sum(axis=0,keepdims=True)
        
        db1 = ((out-y_true) @ self.fc3.T * self.df(y1)) @ self.fc2.T * self.df(x1)
        dw1 = x.T @ db1 
        db1 = db1.sum(axis=0,keepdims=True)
        # import pdb;pdb.set_trace() 
        # dw1 = 
        
        # 返回梯度
        grads = {'fc1': dw1, 'bias1': db1,
                 'fc2': dw2, 'bias2': db2,'fc3': dw3, 'bias3':db3}
        return grads, loss
    
class SGD:
    def __init__(self, lr=0.01, l2=0.001, model=None):
        self.learning_rate = lr
        self.l2 = l2
        self.model = model

    def update(self, grads):
        assert set(grads.keys()) == set(self.model.params.keys()), "参数有误"
        for key in grads.keys():
            # 计算梯度更新部分，包括原始梯度和 L2 正则化项的梯度
            grad_update = grads[key] + 2 * self.l2 * self.model.params[key]
            self.model.params[key] -= self.learning_rate * grad_update

class Dataloader:
    def __init__(self, dataset, batch_size=128):
        assert len(dataset)== 2, "数据格式有误"
        self.data = dataset
        self.total_samples = len(dataset[0])
        self.batch_size = batch_size
        self.num_batches = (len(dataset[0]) + batch_size - 1) // batch_size
        
    def __iter__(self):
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.total_samples)
            # import pdb;pdb.set_trace()
            yield self.data[0][start_idx: end_idx],self.data[1][start_idx: end_idx]
            
    def __len__(self):
        return self.num_batches
            

