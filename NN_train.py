import struct
import torch

from NN import *
from sklearn.metrics import confusion_matrix

# 导入训练集
def load_minist(labels_path, images_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

#读取数据集的函数，MINst数据集本身已经被下载到了本地文件夹内
images, labels = load_minist('train-labels.idx1-ubyte', 'train-images.idx3-ubyte')
test_images, test_labels = load_minist('t10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte')

#对标签值做标准化处理,例如：6转化为[0,0,0,0,0,0,1,0,0,0]
labels = label_binarizer(labels)
test_labels = label_binarizer(test_labels)

#示例话神经网络对象，为输入层，隐藏层，输出层设置初值，设置激活函数为‘logistic’
nn = NeuralNetwork([784, 250, 10], 'logistic')


nn.fit(images, labels)   #调用训练集函数开始训练
# 求混淆矩阵
# 收集测试结果
predictions = []
predictions = nn.predict(images) #调用预测函数求出关于训练集的全部预测结果并存入predictions

#将标签值和预测值全部转为一维，便于confusion_matrix()函数用于计算
predictions=np.argmax(predictions,axis=1)
labels =np.argmax(labels,axis=1)
print(confusion_matrix(labels, predictions))

#找出出错示例
nn.verify(test_images,test_labels) #使用验证集函数开始训练

