import pickle
import os, sys
import numpy as np
import matplotlib.pyplot as plt

"""
手动实现实现神经网络, 用于cifer 10 图片分类
"""

# 读取所有文件夹中的pickle文件，分成训练和测试两个部分
def load_data(path):
    X, Y = [], []
    # 导入pickle文件， 并将每张图片转换成（10000， 32， 32， 3）的矩阵
    def load_picke_file(f_name):
        with open(f_name, 'rb') as f:
            p = pickle.load(f, encoding='latin1')
            x = p['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float') # transpose 为吧channel放在最后
            y = np.array(p['labels']) # Y_train 为10类(0-9), 每个类别5000个, 总共50000张图片类别
            return x, y

    for i in range(1, 6): # batch 1-6
        f_name = os.path.join(path, 'data_batch_%d' % i)
        x, y = load_picke_file(f_name)
        X.append(x)
        Y.append(y)
        X_train = np.concatenate(X)
        Y_train = np.concatenate(Y)
    X_test, Y_test = load_picke_file(os.path.join(path, 'test_batch'))
    return X_train, Y_train, X_test, Y_test


# 展示部分数据集中的图片，每种类型展示7张
def show_picture(Y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples = 7
    for y, c in enumerate(classes):
        idxs = np.flatnonzero(Y_train == y) # 返回值为Y_train array中, 与当前分类y相等的位置, 返回长度为5000, 因为每个类都有5000张图片 
        idxs = np.random.choice(idxs, samples, replace=False) # 随机选7张
        for x, idx in enumerate(idxs):
            plt_idx = x * num_classes + y + 1
            # 1 2 3
            # 4 5 6
            # 7 8 9   subplot(l=3, w=3, p= x * 3 + y + 1)
            plt.subplot(samples, num_classes, plt_idx) # subplot(长点数量, 宽点数量, 点位置)
            plt.imshow(X_train[idx].astype('uint8'))
            if x == 0:
                plt.title(c)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


class DNN:
    def __init__(self,
                hidden_dims, # 为一个list, 长度表示隐藏层层数, 值表示隐藏从维度
                input_dim=3072, # 32 * 32 * 3
                num_classes=10,
                reg=0.3,
                dtype=np.float32):
        # param
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.reg = reg
        self.dtype = dtype
        self.params = {'W':[], 'b':[]}
        D_in, D_out = input_dim, hidden_dims[0]

        # 给每个神经单元做权重初始化
        for i in range(self.num_layers + 1):
            self.params['W'].append(np.random.normal(scale=1e-3, size=(D_in, D_out)).astype(self.dtype))
            self.params['b'].append(np.zeros((D_out, )).astype(self.dtype))
            D_in = D_out # 上一层的输出等于下一层的输入
            if i < (self.num_layers - 1):
                D_out = hidden_dims[i + 1]
            else:
                D_out = num_classes # 最后一层输出为分类个数


    def forward2loss(self, X, Y):
        
        ## FORWARD
        def softmax_loss(X, Y):
            #TODO:完成softmax函数
            N = X.shape[0]
            shifted_X = X - np.max(X, axis=1, keepdims=True) # [b, classes] - [b, max_c] = [b, classes]
            total = np.sum(np.exp(shifted_X), axis=1, keepdims=True) # [b, 1]
            log_probs = shifted_X - np.log(total)
            # print(log_probs.shape)
            probs = np.exp(log_probs)
            loss = - np.sum(log_probs[np.arange(N), Y]) / N
            # print(loss.shape)
            dx = probs.copy()
            dx[np.arange(N), Y] -= 1
            dx /= N
            # print("loss_d_out ", dx.shape)
            return loss, dx

        def relu_forward(X):
            return X * (X > 0) , X # 大于 X > 0 乘1, X <= 0, 乘0

        def nn_forward(X, W, b):
            X_reshape = X.reshape(X.shape[0], -1) # 将 X shape 变换为与 W 相同
            out = X_reshape.dot(W) + b
            return out, (X, W, b)
        
        def layer_forward(X, W, b):
            y, cache_0 = nn_forward(X, W, b)
            out, cache_1 = relu_forward(y)
            return out, (cache_0, cache_1)
        
        ## BACKWARD
        def relu_backward(d_out, cache):
            return d_out * (cache > 0) #反向传播时relu激活  

        def nn_backward(d_out, cache):
            X, W, b = cache #
            # print('====')
            original_shape = X.shape
            X = X.reshape(original_shape[0], -1)
            # 求偏导 (hidden1_out 为loss_fun_in, hidden0_out 为 hidden1_in, 以此类推)
            #      σ(loss)        σ(hidden1_out)    σ(hidden0_out)
            # ---------------- * -------------- * -----------------
            #  σ(hidden1_out)    σ(hidden0_out)       σ(w)
            # print("d_out ", d_out.shape)
            dx = d_out.dot(W.T)
            # print("dx ", dx.shape)
            dw = X.T.dot(d_out)
            # print("dw ", dw.shape)
            db = d_out.T.dot(np.ones((original_shape[0], )))
            # print("db ", db.shape)
            dx = dx.reshape(original_shape)
            # print("dx_up ", dx.shape)
            # print('===')
            return dx, dw, db

        def layer_backward(d_out, cache):
            # cache_0 包含了当前层前向传播时中权重相乘所使用 X, W, b
            # cache_1 包含了当前层前向传播时通过激活函数之前所使用的的 X
            cache_0, cache_1 = cache 
            d_nn = relu_backward(d_out, cache_1)
            dx, dw, db = nn_backward(d_nn, cache_0)
            return dx, dw, db

        X = X.astype(self.dtype)
        out = X
        cache = []

        # 前向传播
        for i in range(self.num_layers): 
            W = self.params['W'][i]
            b = self.params['b'][i]
            # print("W%d: %s" % (i,  W.shape))
            out, c01 = layer_forward(out, W, b)
            # print("out%d: %s" % (i, out.shape))
            cache.append(c01)
        # 最后一层独立计算激活函数, 分开计算, cache 只有c0, 此时, 我们就计算完成了从输入到输出的一次迭代 out
        out, c0 = nn_forward(out, self.params['W'][self.num_layers], self.params['b'][self.num_layers])
        cache.append(c0)
        # 最后一层激活函数为softmax, 且计算loss, d_outshape变化为[b, hidden_dim] -> [b, classes]
        loss, d_out = softmax_loss(out, Y)

        # 反向传播
        # 梯度初始化
        grads = {'W': [0] * (self.num_layers + 1), 'b': [0] * (self.num_layers + 1)}

        # loss + L2 正则化 : loss + 1/2 * learning_rate * W_last 平方和
        i = self.num_layers # 从最后一层开始加L2
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W'][i]))
        # 最后一层反向传播
        d_out, grads['W'][i], grads['b'][i] = nn_backward(d_out, cache.pop()) # cache 抛出最后一个c0
        # 权重更新
        grads['W'][i] += self.reg * self.params['W'][i]

        i -= 1 # 总层数减去最后一层, 更新其他层权重
        for j in range(i, -1, -1):
            d_out, grads['W'][j], grads['b'][j] = layer_backward(d_out, cache.pop())
            loss += 0.5 * self.reg * np.sum(np.square(self.params['W'][j]))
            grads['W'][j] += self.reg * self.params['W'][j]

        return loss, grads, out


    def train(self, train_data, eval_data, iterations, batch_size=1000, learning_rate=1e-3):
        # data
        X_train, Y_train = train_data
        X_eval, Y_eval = eval_data
        self.loss_history = []
        self.train_acc_history = []
        self.eval_acc_history = []
        # iterating & batch
        for i in range(iterations):
            batch_mask = np.random.choice(X_train.shape[0], batch_size)
            X_batch = X_train[batch_mask]
            Y_batch = Y_train[batch_mask]
            loss, grads, scores = self.forward2loss(X_batch, Y_batch)
            # 记录与评测
            # train 
            train_acc = self.evaluate_v2(scores, Y_batch)
            print("train step: %d / %d, loss: %f, accuaracy: %f" %
                        (i, iterations, loss, train_acc))
            self.loss_history.append(loss)
            self.train_acc_history.append(train_acc)
            # evaluate
            if i % batch_size == 0:
                eval_acc = self.evaluate(X_eval, Y_eval)
                print("\n eval step: %d / %d, loss: %f, accuaracy: %f\n" %
                        (i, iterations, loss, eval_acc))
                self.eval_acc_history.append(eval_acc)

            #  参数更新(梯度下降)
            for i in range(self.num_layers + 1):
                self.params['W'][i] -= learning_rate * grads['W'][i]
                self.params['b'][i] -= learning_rate * grads['b'][i]
            
        return self.loss_history


        return 
    def evaluate(self, X, Y):
        _, _, scores = self.forward2loss(X, Y)
        Y_pred = np.argmax(scores, axis=1)
        acc = np.mean(Y_pred == Y)
        return acc

    def evaluate_v2(self, score, Y):
        Y_pred = np.argmax(score, axis=1)
        acc = np.mean(Y_pred == Y)
        return acc



if __name__ == "__main__":

    # wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    # tar -zcvf cifar-10-python.tar.gz

    path = sys.argv[1]
    # load data
    X_train, Y_train, X_test, Y_test = load_data(path)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    # show image
    # show_picture(Y_train)

    X_train_reshape = X_train.reshape(X_train.shape[0], -1)

    # model & train
    dnn = DNN(hidden_dims=[100, 100])
    dnn.train((X_train, Y_train), (X_test, Y_test), iterations=10000)

    # 模型保存
    save_path = 'model.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(dnn, f)

    with open(save_path, 'rb') as f:
        dnn = pickle.load(f)
        print(dnn.loss_history)