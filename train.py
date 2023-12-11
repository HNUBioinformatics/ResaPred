#好像是真实数据和假数据分别放入判别器进行分类，将真实数据正确分类和假数据正确被判别的结果一起进行loss函数计算
#所有蛋白质集合到一个trainloader中
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from CBAM import CBAM
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model_att_test import Conv_att
# from model_test2 import Conv_att

# from win9_1d import Conv_att
from win9_1d_pssm import Conv_att

# from win9_1d_nonpssm import Conv_att

# from inception_conv1d import Conv_att
# from inception_test import Conv_att

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# features_path="/home/u2208283057/resnet_selfatt/model/train_datasets_feature_nonstrict.pickle"
# bfactor_path="/home/u2208283057/resnet_selfatt/model/train_datasets_bfactor_nonstrict.pickle"


# datasets_path="/home/u2208283057/resnet/pssm"
#
# mode_para_path="/home/u2208283057/resnet/model/model_parameter"

# datasets_path="/home/u2208283057/resnet/8977_datasets_nonpssm"


# features_path="D:/MachineLearning/蛋白质/蛋白质柔性/大型正式训练/data/features_归一化"
# bfactor_path="D:/MachineLearning/蛋白质/蛋白质柔性/大型正式训练/data/Bfactor_labels"
# mode_para_path="D:/MachineLearning/蛋白质/蛋白质柔性/大型正式训练/model/model_parameter/第一折"
# results_path="D:/MachineLearning/蛋白质/蛋白质柔性/大型正式训练/model/results"
# datasets_path="D:/MachineLearning/蛋白质/蛋白质柔性/大型正式训练/2084_datasets/7特征"


datasets_path="/root/autodl-tmp/resnet/8977_pssm"
ns_bfactor="/root/autodl-tmp/resnet/8977_9win2d"
s_bfactor="/root/autodl-tmp/resnet/8977_9win2d"
mode_para_path="/root/autodl-tmp/resnet/nonstrict_model_parameter"
# datasets_path="/root/autodl-tmp/resnet/8977_9win2d"
# datasets_path="/root/autodl-tmp/resnet/2084_9win1d"


# datasets_path="/home/u2208283057/resnet/data/7features"
# nonwin_path="/home/u2208283057/resnet"
# mode_para_path="/home/u2208283057/resnet/nonstrict_model_parameter"
# device="cpu"



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def get_file(strict_class):
#     features=[]
#     bfactor_labels=[]
#     # for i in range(10):
#     #     features[i]=[]
#     #     bfactor_labels[i]=[]
#     for i in range(10):
#         filename1 = nonwin_path + '/' + strict_class + '/fold' + str(i)  # 验证集对应的文件名
#         # filename = datasets_path + '/blance_' + strict_class + '/fold' + str(i)  # 验证集对应的文件名
#         filename = datasets_path+ '/' + strict_class + '/fold' + str(i)  # 验证集对应的文件名
#
#         with open(filename + "_feature_" + strict_class + ".pickle", "rb") as file1:
#             features.append(pickle.load(file1))
#         with open(filename + "_bfactor_" + strict_class + ".pickle", "rb") as file2:
#             bfactor_labels.append(pickle.load(file2))
#     return features,bfactor_labels





def get_evaluation(y_true,y_pred):
    tp = torch.sum((y_true == 1) & (y_pred == 1))
    tn = torch.sum((y_true == 0) & (y_pred == 0))
    fp = torch.sum((y_true == 0) & (y_pred == 1))
    fn = torch.sum((y_true == 1) & (y_pred == 0))

    # print("tp,tn,fp,fn:",tp,tn,fp,fn)
    # 计算 TP、TN、FP、FN
    # 计算准确率
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) != 0 else torch.tensor(0)
    recall=tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(0)
    # 计算F1值
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else torch.tensor(0)
    # 计算敏感度（召回率）
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(0)
    # 计算MCC
    mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else torch.tensor(0)
    # print("precision,recall,f1,sensitivity,mcc",precision,recall,f1,sensitivity,mcc)
    return precision.item(),f1.item(),sensitivity.item(),mcc.item()




# #优化器
model=Conv_att(1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999))
# optimizer = torch.optim.SGD(params=[w], lr=0.1, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(model,train_loader,criterion,optimizer,epoch,datasets_num):
    model.train()
    train_loss=0
    train_acc=0
    batch=0#当前batch的ID
    for count,datas in enumerate(train_loader):
        batch+=1
        optimizer.zero_grad()
        feature, label = datas
        batch_size = len(train_loader)
        # 将标签设置为float型，为了与sigmoid输出计算损失
        feature=feature.float()
        # label=label.float()
        feature = feature.to(device)
        label = label.to(device)
        # label = torch.tensor(label, dtype=torch.float)
        label_onehot = torch.eye(2)[label.long(), :]#独热编码，保证标签为二维数据
        label_onehot = label_onehot.to(device)

        outputs = model(feature)
        # print(outputs.shape,outputs)
        loss = criterion(outputs, label_onehot)
        # print(outputs.shape,outputs.dtype,labels.shape,labels.dtype)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # outputs_class = np.argmax(outputs, axis=1)
        y_pred=outputs.argmax(1)
        #评价指标的计算
        accuracy_num = (y_pred == label).sum()
        accuracy=accuracy_num*100/len(feature)
        train_acc = train_acc + accuracy  # 计算每一个batch中正确分类真实数据标签的个数
        # precision,f1,sensitivity,mcc = get_evaluation(label, y_pred)
        progress=batch*100/batch_size
        # print(batch,batch_size)

        time.sleep(0.001)
        print(f"\rEpoch:{epoch:d} 第{datasets_num}个数据集 batch :{batch}| Train Loss: {loss.item():.4f} Train Acc: {accuracy:.2f}% |progress: {progress:.2f}%",end='',  flush=True)

        # print(f"\rEpoch:{epoch:d} 第{datasets_num}个数据集 batch :{batch} device:{feature.device} | Train Loss: {loss.item():.4f} | Train Acc: {accuracy:.2f}% |progress: {progress:.2f}%",end='',  flush=True)

    return train_loss/batch_size,train_acc/batch_size




def test(model,train_loader):
    model.eval()
    train_loss = 0
    train_acc = 0
    total_precision = 0
    total_f1 = 0
    total_sensitivity = 0
    total_mcc = 0
    batch = 0  # 当前batch的ID
    with torch.no_grad():
        for count, datas in enumerate(train_loader):
            batch += 1
            feature, label = datas
            batch_size = len(train_loader)
            # 将标签设置为float型，为了与sigmoid输出计算损失
            feature = feature.float()
            label = label.float()
            feature = feature.to(device)
            label = label.to(device)
            # label = torch.tensor(label, dtype=torch.float)

            label_onehot = torch.eye(2)[label.long(), :]  # 独热编码，保证标签为二维数据
            label_onehot = label_onehot.to(device)

            # label = torch.unsqueeze(label, dim=1)  # 将标签维度设置为（10,1）
            # print(feature.shape, label.shape)
            outputs = model(feature)
            # print(outputs.shape,outputs)
            loss = criterion(outputs, label_onehot)
            train_loss += loss.item()

            # outputs_class = np.argmax(outputs, axis=1)
            y_pred = outputs.argmax(1)
            # 评价指标的计算
            accuracy_num = (y_pred == label).sum()# 计算每一个batch中正确分类真实数据标签的个数
            accuracy = accuracy_num * 100 / len(feature)
            train_acc = train_acc + accuracy
            precision, f1, sensitivity, mcc = get_evaluation(label,y_pred)
            # progress = batch * 100 / batch_size
            # time.sleep(0.001)
            # print(
            #     f"\r Test batch :{batch} |  loss: {loss:.4f} |  Acc: {accuracy:.2f}% |progress: {progress:.2f}%",
            #     end='', flush=True)

            total_precision = total_precision + precision
            total_f1 = total_f1 + f1
            total_sensitivity = total_sensitivity + sensitivity
            total_mcc = total_mcc + mcc
    precision = total_precision/batch_size
    f1 = total_f1/batch_size
    sensitivity = total_sensitivity/batch_size
    mcc = total_mcc/batch_size

    # print(f"batch :{batch}  Train Loss: {train_loss/ len(train_loader):.4f} | Train Acc: {train_acc/ len(train_loader) :.2f}%")
    # , total_precision / batch_size, total_f1 / batch_size, total_sensitivity / batch_size, total_mcc / batch_size
    return train_loss / batch_size, train_acc / batch_size,precision, f1, sensitivity, mcc


def model_val(epoch,batchsize,strict_class,worker_num,val_num,feature_length):
    #加载验证数据集
    filename = datasets_path +'/'+strict_class+ '/fold' + str(val_num)#验证集对应的文件名

    with open(filename + "_feature_"+strict_class+".pickle", "rb") as file1:
        feature = pickle.load(file1)
    with open(filename + "_bfactor_"+strict_class+".pickle", "rb") as file2:
        bfactor_labels = pickle.load(file2)
    features = torch.tensor(feature)
    features = features.reshape(len(features), feature_length * 9)
    features = torch.unsqueeze(features, dim=1)
    factor_labels = torch.tensor(bfactor_labels, dtype=int)
    # print(len(features), len(factor_labels))
    dataset = TensorDataset(features, factor_labels)
    shuffle = True
    val_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,num_workers=worker_num)
    # 验证集上的测试效果
    model.load_state_dict(torch.load('%s/%d测%s_model_epoch_%d.pth' % (mode_para_path,(val_num+1),strict_class, epoch + 1)))
    model.eval()
    train_loss = 0
    train_acc = 0
    total_precision = 0
    total_f1 = 0
    total_sensitivity = 0
    total_mcc = 0
    batch = 0  # 当前batch的ID
    with torch.no_grad():
        for count, datas in enumerate(val_loader):
            batch += 1
            feature, label = datas
            batch_size = len(val_loader)
            # 将标签设置为float型，为了与sigmoid输出计算损失
            feature = feature.float()
            label = label.float()
            feature = feature.to(device)
            label = label.to(device)
            # label = torch.tensor(label, dtype=torch.float)

            label_onehot = torch.eye(2)[label.long(), :]  # 独热编码，保证标签为二维数据
            label_onehot = label_onehot.to(device)

            # label = torch.unsqueeze(label, dim=1)  # 将标签维度设置为（10,1）
            # print(feature.shape, label.shape)
            outputs = model(feature)
            # print(outputs.shape,outputs)
            loss = criterion(outputs, label_onehot)
            train_loss += loss.item()

            # outputs_class = np.argmax(outputs, axis=1)
            y_pred = outputs.argmax(1)
            # 评价指标的计算
            accuracy_num = (y_pred == label).sum()# 计算每一个batch中正确分类真实数据标签的个数
            accuracy = accuracy_num * 100 / len(feature)
            train_acc = train_acc + accuracy
            precision, f1, sensitivity, mcc = get_evaluation(label,y_pred)
            # progress = batch * 100 / batch_size
            # print(batch,batch_size)
            #
            # time.sleep(0.001)
            # print(
            #     f"\r Val batch :{batch} device:{feature.device} | Val loss: {loss:.2f} | Val Acc: {accuracy:.2f}% |progress: {progress:.2f}%",
            #     end='', flush=True)

            total_precision = total_precision + precision
            total_f1 = total_f1 + f1
            total_sensitivity = total_sensitivity + sensitivity
            total_mcc = total_mcc + mcc
    precision = total_precision/batch_size
    f1 = total_f1/batch_size
    sensitivity = total_sensitivity/batch_size
    mcc = total_mcc/batch_size

    # print(f"batch :{batch}  Train Loss: {train_loss/ len(train_loader):.4f} | Train Acc: {train_acc/ len(train_loader) :.2f}%")
    # , total_precision / batch_size, total_f1 / batch_size, total_sensitivity / batch_size, total_mcc / batch_size
    return train_loss / batch_size, train_acc / batch_size,precision, f1, sensitivity, mcc




# def model_train(train_feature,train_label,strict_class,worker_num):
#     batchsize = 80
#     epochs = 10  # 训练的epoch次数
#     best_id=0
#     lower_loss=10
#     for epoch in range(epochs):
#         start = time.time()
#         features = torch.tensor(train_feature)
#         features = torch.unsqueeze(features, dim=1)
#         factor_labels = torch.tensor(train_label, dtype=int)
#             # print(len(features), len(factor_labels))
#         dataset = TensorDataset(features, factor_labels)
#         shuffle = True
#         train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,num_workers=worker_num)
#         loss_, acc= train(model, train_loader, criterion, optimizer, epoch + 1)
#         end = time.time()
#         print(f"\nEPOCH:{epoch + 1} {strict_class} | Train Loss: {loss_:.4f} | Train Acc: {acc :.2f}%  用时:{end-start:.4f}")
#         # 模型参数保存
#         torch.save(model.state_dict(), '%s/%s_model_epoch_%d.pth' % (mode_para_path,strict_class, epoch + 1))
#         val_loss, val_acc,precision, f1, sensitivity, mcc=model_val(features_dic,bfactor_labels_dic,epoch,batchsize,strict_class,worker_num)
#         print(f"EPOCH:{epoch + 1} {strict_class} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc :.2f}% precision:{precision} f1:{f1} sensitivity:{sensitivity} mcc:{mcc}")
#         if val_loss<lower_loss:
#             lower_loss=val_loss
#             best_id=epoch+1
#     print(strict_class,best_id)





def model_test(pth_path,strict_class,num,feature_length):
    model.load_state_dict(torch.load(pth_path))
    batchsize = 80
    shuffle = True
    filename = datasets_path + '/' + strict_class + '/fold' + str(num)  # 对应的文件名
    with open(filename + "_feature_" + strict_class + ".pickle", "rb") as file1:
        features = pickle.load(file1)
    with open(filename + "_bfactor_" + strict_class + ".pickle", "rb") as file2:
        bfactor_labels = pickle.load(file2)
    features = torch.tensor(features)
    features = features.reshape(len(features), feature_length * 9)
    features = torch.unsqueeze(features, dim=1)
    factor_labels = torch.tensor(bfactor_labels, dtype=int)
    # print(len(features), len(factor_labels))
    dataset = TensorDataset(features, factor_labels)
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle)
    loss, acc, precision, f1, sensitivity, mcc = test(model, train_loader)
    print(f"Test {strict_class} test_num{num} loss: {loss :.4f} Test Acc: {acc :.2f}% precision:{precision} f1:{f1} sensitivity:{sensitivity} mcc:{mcc}")
    return acc, precision, f1, sensitivity, mcc

def model_train(train_num,strict_class,worker_num,val_num,feature_length):
    batchsize = 80
    epochs = 5  # 训练的epoch次数
    best_id=0
    lower_loss=10
    for epoch in range(epochs):
        start = time.time()
        total_loss=0
        total_acc=0

        for i in train_num:
            filename = datasets_path + '/' + strict_class + '/fold' + str(i)  # 验证集对应的文件名
            with open(filename + "_feature_" + strict_class + ".pickle", "rb") as file1:
                train_feature = pickle.load(file1)
            with open(filename + "_bfactor_" + strict_class + ".pickle", "rb") as file2:
                train_label = pickle.load(file2)
            features = torch.tensor(train_feature)
            features=features.reshape(len(features),feature_length*9)
            features = torch.unsqueeze(features, dim=1)
            factor_labels = torch.tensor(train_label, dtype=int)
            # print(len(features), len(factor_labels))
            dataset = TensorDataset(features, factor_labels)
            shuffle = True
            train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=worker_num)
            loss_, acc = train(model, train_loader, criterion, optimizer, epoch + 1,i)
            total_loss=total_loss+loss_
            total_acc=total_acc+acc
        loss=total_loss/len(train_num)
        acc=total_acc/len(train_num)
        end = time.time()
        print(f"\nEPOCH:{epoch + 1} {strict_class} | Train Loss: {loss:.4f} | Train Acc: {acc :.2f}%  用时:{end-start:.4f}")
        # 模型参数保存
        torch.save(model.state_dict(), '%s/%d测%s_model_epoch_%d.pth' % (mode_para_path,(val_num+1),strict_class, epoch + 1))
        val_loss, val_acc,precision, f1, sensitivity, mcc=model_val(epoch,batchsize,strict_class,worker_num,val_num,feature_length)
        print(f"EPOCH:{epoch + 1} {strict_class} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc :.2f}% precision:{precision} f1:{f1} sensitivity:{sensitivity} mcc:{mcc}")
        if val_loss<lower_loss:
            lower_loss=val_loss
            best_id=epoch+1
    print(strict_class,(val_num+1),best_id)



def add_all(all_acc, all_precision, all_f1, all_sensitivity, all_mcc,acc, precision, f1, sensitivity, mcc):
    all_acc=all_acc+acc
    all_precision=all_precision+precision
    all_f1=all_f1+f1
    all_sensitivity=all_sensitivity+sensitivity
    all_mcc=all_mcc+mcc
    return all_acc, all_precision, all_f1, all_sensitivity, all_mcc


if __name__ == '__main__':
    # train_num=[0,1,2,3,4,5,6,7]#训练数据集所包括的子集对应的数字
    # train_feature=[]
    # train_label=[]
    # for i in train_num:
    #     train_feature=train_feature+features_dic[i]
    #     train_label=train_label+bfactor_labels_dic[i]
    # model_train(features_dic,bfactor_labels_dic,strict_class="nonstrict")
    feature_len=20

    # train_num = [2,3,4,5,6,7,8,9]  # 训练数据集所包括的子集对应的数字 01
    # model_train(train_num,strict_class="nonstrict",worker_num=4,val_num=0,feature_length=feature_len)
    # model_train(train_num,strict_class="strict",worker_num=8,val_num=0,feature_length=feature_len)
    #
    # train_num = [0,3,4,5,6,7,8,9]  # 训练数据集所包括的子集对应的数字  12
    # model_train(train_num,strict_class="nonstrict",worker_num=4,val_num=1,feature_length=feature_len)
    # model_train(train_num,strict_class="strict",worker_num=4,val_num=1,feature_length=feature_len)

    # train_num = [0,1,4,5,6,7,8,9]  # 训练数据集所包括的子集对应的数字 23
    # model_train(train_num,strict_class="nonstrict",worker_num=4,val_num=2,feature_length=feature_len)
    # model_train(train_num,strict_class="strict",worker_num=4,val_num=2,feature_length=feature_len)
    #
    train_num = [0,1,2,5,6,7,8,9]  # 训练数据集所包括的子集对应的数字 34
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=3,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=3,feature_length=feature_len)

    train_num = [0,1,2,3,6,7,8,9]  # 训练数据集所包括的子集对应的数字 45
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=4,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=4,feature_length=feature_len)

    train_num = [0,1,2,3,4,7,8,9]  # 训练数据集所包括的子集对应的数字 56
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=5,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=5,feature_length=feature_len)

    train_num = [0,1,2,3,4,5,8,9]  # 训练数据集所包括的子集对应的数字 67
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=6,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=6,feature_length=feature_len)

    train_num = [0,1,2,3,4,5,6,9]  # 训练数据集所包括的子集对应的数字 78
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=7,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=7,feature_length=feature_len)

    train_num = [0,1,2,3,4,5,6,7]  # 训练数据集所包括的子集对应的数字 89
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=8,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=8,feature_length=feature_len)
    #
    train_num = [1,2,3,4,5,6,7,8]  # 训练数据集所包括的子集对应的数字 90
    model_train(train_num,strict_class="nonstrict",worker_num=8,val_num=9,feature_length=feature_len)
    model_train(train_num,strict_class="strict",worker_num=8,val_num=9,feature_length=feature_len)



#测试模块，不要删

    # all_ns_acc=0
    # all_ns_precision=0
    # all_ns_f1=0
    # all_ns_sensitivity=0
    # all_ns_mcc=0
    # all_s_acc=0
    # all_s_precision=0
    # all_s_f1=0
    # all_s_sensitivity=0
    # all_s_mcc=0
    # len_pth=10
    # for i in range(10):
    #     test_pth = mode_para_path + "/pssm_aa_"+str(i)+"_nonstrict.pth"
    #     ns_acc, ns_precision, ns_f1, ns_sensitivity, ns_mcc = model_test(test_pth, "nonstrict", i,feature_length=feature_len)
    #     test_pth = mode_para_path + "/pssm_aa_"+str(i)+"_strict.pth"
    #     s_acc, s_precision, s_f1, s_sensitivity, s_mcc = model_test(test_pth, "strict", i, feature_length=feature_len)
    #     all_ns_acc,all_ns_precision,all_ns_f1,all_ns_sensitivity,all_ns_mcc=add_all(all_ns_acc,all_ns_precision,all_ns_f1,all_ns_sensitivity,all_ns_mcc,ns_acc, ns_precision, ns_f1, ns_sensitivity, ns_mcc)
    #     all_s_acc,all_s_precision,all_s_f1,all_s_sensitivity,all_s_mcc=add_all(all_s_acc,all_s_precision,all_s_f1,all_s_sensitivity,all_s_mcc,ns_acc, s_precision, s_f1, s_sensitivity, s_mcc)
    #
    # all_ns_acc=all_ns_acc/len_pth
    # all_ns_precision=all_ns_precision/len_pth
    # all_ns_f1=all_ns_f1/len_pth
    # all_ns_sensitivity=all_ns_sensitivity/len_pth
    # all_ns_mcc=all_ns_mcc/len_pth
    #
    # all_s_acc=all_s_acc/len_pth
    # all_s_precision=all_s_precision/len_pth
    # all_s_f1=all_s_f1/len_pth
    # all_s_sensitivity=all_s_sensitivity/len_pth
    # all_s_mcc=all_s_mcc/len_pth
    # print(f"nonstrict: Acc: {all_ns_acc :.2f}% precision:{all_ns_precision} f1:{all_ns_f1} sensitivity:{all_ns_sensitivity} mcc:{all_ns_mcc}")
    # print(f"strict: Acc: {all_s_acc :.2f}% precision:{all_s_precision} f1:{all_s_f1} sensitivity:{all_s_sensitivity} mcc:{all_s_mcc}")

#结束分隔符

    #
    # test_pth=mode_para_path+"/nonss_1_nonstrict.pth"
    # ns_acc, ns_precision, ns_f1, ns_sensitivity, ns_mcc=model_test(test_pth,"nonstrict",1,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_1_strict.pth"
    # model_test(test_pth,"strict",1,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_2_nonstrict.pth"
    # model_test(test_pth,"nonstrict",2,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_2_strict.pth"
    # model_test(test_pth,"strict",2,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_3_nonstrict.pth"
    # model_test(test_pth,"nonstrict",3,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_3_strict.pth"
    # model_test(test_pth,"strict",3,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_4_nonstrict.pth"
    # model_test(test_pth,"nonstrict",4,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_4_strict.pth"
    # model_test(test_pth,"strict",4,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_5_nonstrict.pth"
    # model_test(test_pth,"nonstrict",5,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_5_strict.pth"
    # model_test(test_pth,"strict",5,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_6_nonstrict.pth"
    # model_test(test_pth,"nonstrict",6,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_6_strict.pth"
    # model_test(test_pth,"strict",6,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_7_nonstrict.pth"
    # model_test(test_pth,"nonstrict",7,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_7_strict.pth"
    # model_test(test_pth,"strict",7,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_8_nonstrict.pth"
    # model_test(test_pth,"nonstrict",8,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_8_strict.pth"
    # model_test(test_pth,"strict",8,feature_length=feature_len)
    #
    # test_pth=mode_para_path+"/nonss_9_nonstrict.pth"
    # model_test(test_pth,"nonstrict",9,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonss_9_strict.pth"
    # model_test(test_pth,"strict",9,feature_length=feature_len)











    # test_pth=mode_para_path+"/nonaaindex_2_nonstrict.pth"
    # model_test(test_pth,"nonstrict",2,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonaaindex_2_strict.pth"
    # model_test(test_pth,"strict",2,feature_length=feature_len)






    # test_pth=mode_para_path+"/nonstructure_9_nonstrict.pth"
    # model_test(test_pth,"nonstrict",9,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonstructure_9_strict.pth"
    # model_test(test_pth,"strict",9,feature_length=feature_len)






    # test_pth="nonangle_9_nonstrict.pth"
    # model_test(test_pth,"nonstrict",9,feature_length=feature_len)
    # test_pth="nonangle_9_strict.pth"
    # model_test(test_pth,"strict",9,feature_length=feature_len)







    # test_pth=mode_para_path+"/nononehot_9_nonstrict.pth"
    # model_test(test_pth,"nonstrict",9,feature_length=feature_len)
    # test_pth=mode_para_path+"/nononehot_9_strict.pth"
    # model_test(test_pth,"strict",9,feature_length=feature_len)




    # feature_len=101
    # test_pth=mode_para_path+"/nonpssm_5_nonstrict.pth"
    # model_test(test_pth,"nonstrict",5,feature_length=feature_len)
    # test_pth=mode_para_path+"/nonpssm_5_strict.pth"
    # model_test(test_pth,"strict",5,feature_length=feature_len)















    #
    # features_dic, bfactor_labels_dic = get_file("strict")
    # # train_num=[0,1,2,3,4,5,6,7]#训练数据集所包括的子集对应的数字
    # # train_feature=[]
    # # train_label=[]
    # # for i in train_num:
    # #     train_feature=train_feature+features_dic[i]
    # #     train_label=train_label+bfactor_labels_dic[i]
    # model_train(features_dic,bfactor_labels_dic,strict_class="strict",worker_num=8)







