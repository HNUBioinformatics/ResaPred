#好像是真实数据和假数据分别放入判别器进行分类，将真实数据正确分类和假数据正确被判别的结果一起进行loss函数计算
#所有蛋白质集合到一个trainloader中
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from win9_1d import Conv_att
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径
mode_para_path = os.path.join(current_dir, "model_parameter")
datasets_path = os.path.join(current_dir, "data", "win9_2d")
ns_bfactor = os.path.join(current_dir, "data", "win9_2d")
s_bfactor = os.path.join(current_dir, "data", "win9_2d")
output_folder = os.path.join(current_dir, "model_output")

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    y_pred_list=[]
    label_list=[]
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
            # print(label)
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

            y_pred=y_pred.tolist()
            y_pred_list=y_pred_list+y_pred
            label = label.int()
            label=label.tolist()
            label_list=label_list+label
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
    return label_list,y_pred_list,train_loss / batch_size, train_acc / batch_size,precision, f1, sensitivity, mcc


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




def model_test(protein_id,pth_path,strict_class,feature_length):
    model.load_state_dict(torch.load(pth_path))
    batchsize = 35
    shuffle = False
    filename = datasets_path + '/' + strict_class +'/' +protein_id  # 对应的文件名
    with open(filename +'_'+ strict_class + "_features" + ".pickle", "rb") as file1:
        features = pickle.load(file1)
    with open(filename +'_' + strict_class+ "_bfactor" + ".pickle", "rb") as file2:
        bfactor_labels = pickle.load(file2)
    features = torch.tensor(features)
    features = features.reshape(len(features), feature_length * 9)
    features = torch.unsqueeze(features, dim=1)
    factor_labels = torch.tensor(bfactor_labels, dtype=int)
    # print(len(features), len(factor_labels))
    dataset = TensorDataset(features, factor_labels)
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle)
    label_list,y_pred_list,loss, acc, precision, f1, sensitivity, mcc = test(model, train_loader)
    print("预测标签")
    print(len(y_pred_list),y_pred_list)
    print("原本标签")
    print(len(label_list),label_list)

    # with open(datasets_path+"/结果/"+protein_id+'_'+strict_class+".txt","w") as file3:
    #     file3.write(y_pred_list)
    print(f"Test {strict_class} loss: {loss :.4f} Test Acc: {acc :.2f}% precision:{precision} f1:{f1} sensitivity:{sensitivity} mcc:{mcc}")
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

    feature_len=109
    with open("example.fasta","r") as fasta_file:
    for line in fasta_file:
        if line.startswith(">"):
            protein_id=line[1:6]
            print(protein_id)
            test_pth = "nonstrict.pth"
            ns_acc, ns_precision, ns_f1, ns_sensitivity, ns_mcc = model_test(protein_id,test_pth, "nonstrict", feature_length=feature_len)
            test_pth = "strict.pth"
            s_acc, s_precision, s_f1, s_sensitivity, s_mcc = model_test(protein_id,test_pth, "strict", feature_length=feature_len)









    #
    # features_dic, bfactor_labels_dic = get_file("strict")
    # # train_num=[0,1,2,3,4,5,6,7]#训练数据集所包括的子集对应的数字
    # # train_feature=[]
    # # train_label=[]
    # # for i in train_num:
    # #     train_feature=train_feature+features_dic[i]
    # #     train_label=train_label+bfactor_labels_dic[i]
    # model_train(features_dic,bfactor_labels_dic,strict_class="strict",worker_num=8)







