#y = 1.447x+0.089+e
import numpy as np


data = []#保存样本集的列表
for i in range(100):#循环采样100个点
    x = np.random.uniform(-10.,10.)#随机采样输入x
    #采样高斯噪声
    eps = np.random.normal(0.,0.1)
    #得到模型的输出
    y = 1.477*x+0.089+eps
    data.append([x,y])#保存样本点
data = np.array(data)#转换为2D Numpy数组

#根据当前的w，b参数计算均方差损失
def mse(w,b,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y-(w*x+b))**2
    #将累加的误差求平均，得到均方差
    return totalError/float(len(points))

#计算梯度
def step_gradient(b_current,w_current,points, lr):
    # 计算误差函数在所有点上的导数，并更新w,b
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/M)*(w_current*x+b_current-y)
        w_gradient += (2/M)*(w_current*x+b_current-y)*x
    #根据梯度下降算法更新w，b，其中lr为学习率
    new_b = b_current - (lr*b_gradient)
    new_w = w_current - (lr*w_gradient)
    return [new_b,new_w]

def gradient_decent(points,starting_b,starting_w,lr,num_iterations):
    b = starting_b
    w = starting_w
    #计算梯度并更新一次
    for step in range(num_iterations):
        #计算梯度并更新一次
        b,w = step_gradient(b,w,np.array(points),lr)
        loss = mse(w,b,points)#计算当前的均方差，用于监控训练进度
        if step%50 == 0:#打印误差和实时的w，b值
            print(f"iteration:{step},loss:{loss},w:{w},b:{b}")
    return [b,w]#返回最后一次的w，b值

def main():
    #加载训练数据，数据是通过真实模型添加观测误差采样得到的
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    #训练优化1000次，返回最优w,b和驯良Loss的下降过程
    [b,w] = gradient_decent(data,initial_b,initial_w,lr,num_iterations)
    loss = mse(w,b,data)
    print(f'Final loss: {loss}, w:{w}, b:{b}')

if __name__ == "__main__":
    main()