import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch

'''自建包'''
import _YSZ_ as ysz
from _YSZ_ import *

'''超参数'''
param = {
    'test_batch_size': 10,
    'epsilon': 0.02,
}

'''加载测试数据'''
test_dataset = ysz.Data_Reader.Cifar10.Cifar10_dataset().get_test_dataset()
loader_test = Data_Reader.get_dataloader(dataset=test_dataset, batch_size=param['test_batch_size'], shuffle=False)

'''加载模型'''
# modelpath='./train1_clear.pkl'
# modelpath='./train2_advT.pkl'
#modelpath='./train4_AdvT_InputZ2D.pkl'
modelpath='./train3_AdvT_InputZ3D.pkl'

net = ysz.Models.LeNet5.Lenet5_cifar10()
net = Models.load_state_dict(net, modelpath)
Transformer.enable_cuda(net)  # 使用cuda
# num_correct, num_samples, acc = Optimizer.test(net, loader_test)
# print('[Start] right predict:(%d/%d) ,pre test_acc=%.4f%%' % (num_correct, num_samples, acc))

'''模型测试过程'''
net.eval()  # 推断模式
for p in net.parameters():  # 将模型参数的梯度获取设为false（）
    p.requires_grad = False
Optimizer.test(net, loader_test)  # 测试干净样本的性能

'''对抗样本攻击'''
adversary = ysz.Adversary.FGSM.FGSM(net, param['epsilon'])  # 攻击方法
#adversary = ysz.Adversary.LinfPGD.LinfPGDAttack(net, param['epsilon'])


#t0 = time()  # 当前时间
total_correct = 0
total_samples = 0
for t, (X, y) in enumerate(loader_test):
    y_pred = Optimizer.pred_batch(X, net)  # 获得干净样本的结果
    # print(torch.min(X))
    # print(torch.max(X))
    X_adv = adversary.perturb(X.numpy(), y)  # 加入扰动，生成对抗样本
    #noiseGen = Tools.ImageProcessing.Noise.Noise()
    #X_adv_noise = noiseGen.add_noise(X_adv, 0.1)

    X_input = X_adv
    y_pred_adv = Optimizer.pred_batch(X_input, net)  # 预测

    total_correct += (y_pred_adv.numpy() == y.numpy()).sum()  # 正确的个数
    total_samples += param['test_batch_size']  # 总个数

    if total_samples % 100 == 0:  # 输出结果
        acc = total_correct / total_samples
        print('Got %d/%d correct (%.2f%%) on the perturbed data'
              % (total_correct, total_samples, 100 * acc))


        # 展示图片
        '''
    if t == 0:
        img1 = to_pil_image(X_adv[0])
        img2 = to_pil_image(X[0])
        for i in range(5):  # 第一行为干净图片
            plt.subplot(3, 5, i + 1)
            plt.imshow(to_pil_image(X[i]))
        for i in range(5):  # 第二行为对抗样本
            plt.subplot(3, 5, 5 + i + 1)
            plt.imshow(to_pil_image(X_adv[i]))
        for i in range(5):  # 第二行为对抗样本
            plt.subplot(3, 5, 10 + i + 1)
            plt.imshow(to_pil_image(X[i]))
        plt.ioff()
        plt.show()
        '''

    # total_samples = len(loader_test.dataset)
    # acc = total_correct / total_samples
    # print('Got %d/%d correct (%.2f%%) on the perturbed data'
    #       % (total_correct, total_samples, 100 * acc))

