import torch
import numpy as np
"""
上课tensor讲解内容
GitHubCode:
https://github.com/JackZhang9/torch_learn_tensor/blob/master/20230223/%E5%BC%A0%E9%87%8F%E7%9A%84%E8%BF%90%E7%AE%97.py
https://github.com/imangoa/pytorch_learning/blob/main/%E5%88%9B%E5%BB%BAtensor.ipynb
"""

'''torch里面的变量是tensor，tensor的创建'''
a=torch.rand((3,4))  # 创建符合均匀分布的随机的张量
print('0-1之间均匀分布\n{}'.format(a))

a1=torch.rand_like(a)  # 创建和a类似的张量
print('类似的分布\n{}'.format(a1))

a2=torch.randn((3,4)) # 创建符合正态分布的随机的张量
print('正态分布\n{}'.format(a2))

'''创建固定张量'''
a3=torch.zeros((3,4))  # 创建全0张量
print('全0张量\n{}'.format(a3))

a4=torch.ones((3,4))  # 创建全1张量
print('全1张量\n{}'.format(a4))

a5=torch.zeros_like(a2)  # 创建形状类似的全0张量
print('形状类似全0\n{}'.format(a5))

a6=torch.ones_like(a2)  # 创建形状类似的全1张量
print('形状类似全1\n{}'.format(a6))

a7=torch.Tensor([[1,2],[2,3]])
print('创建简单张量\n{}'.format(a7))

np_array=np.random.randint(0,100,(3,2)) # NumPy 库生成一个 3x2 的二维数组，数组中的元素是随机整数，取值范围在 0 到 99 之间（左闭右开区间）
a8=torch.from_numpy(np_array)  # 从numpy数组创建张量
print('从numpy数组创建\n{}'.format(a8))

a9=torch.arange(1,10)
print('一个一维张量\n{}'.format(a9))

a10=torch.linspace(0,10,5)
print('等差张量\n{}'.format(a10))

a11=torch.logspace(0,2,2)
print('等比张量\n{}'.format(a11))

a12=torch.eye(3,3)  # 创建一个指定形状的对角张量
print('对角张量\n{}'.format(a12))

a13=torch.randint(0,100,(3,4)) # 创建指定形状的随机整数张量
print('指定形状的随机整数张量\n{}'.format(a13))

a14=a13.resize(2,6)
print('变形状\n{}'.format(a14))

a15=a13.reshape(6,2)  #
print('reshape\n{}'.format(a15))

a16=a13.view(4,3)
print('变形状\n{}'.format(a16))

a17=a13.numpy()
print('转=numpy{}\n{}'.format(type(a17),a17))

# a18=a13.cuda()  # 放到gpu  ,易出错
# print('{}'.format(a18))


'''张量的自动求导'''
x=torch.rand((2,3))
w=torch.rand((3,4),requires_grad=True) #requires_grad=True表示梯度跟踪
b=torch.rand((4),requires_grad=True)
# print('a{}'.format(a))
# print('b{}'.format(b))
# print('c{}'.format(c))

'''前向传播'''
d1=torch.matmul(x,w)+b
print('d1',d1)
loss1=torch.relu(d1)  # 激活函数
loss1=torch.sum(loss1)
print('loss',loss1)

'''反向传播'''
loss1.backward()
for i in range(10):
    lr=0.01
    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)
    print('{}轮新的w的grad={}'.format(i,w))  #  optimizer.step()
    print('{}轮新的b的grad={}'.format(i,b))


'''张量的基本运算，加减乘，转置'''
a=torch.randn((2,3))
b=torch.randint(0,100,(3,2))
c=torch.randint(0,50,(3,2))
'''张量乘法'''
mul=torch.matmul(a,b.to(torch.float32))  # 用.to(torch.float32)转化为float类型数值
print('{}*{}={}'.format(a.shape,b.shape,mul.shape))
print('mul={}'.format(mul))

'''张量加法'''
ad=torch.add(b,c)
print('加法={}'.format(ad))

'''张量减法'''
div=torch.div(b,c)
print('减法={}'.format(div))


'''张量的转置'''
transpo=mul.T
print('转置\n{}'.format(transpo))

transpo1=torch.transpose(mul,1,0)
print('转置2\n{}'.format(transpo1))