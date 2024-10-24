import pickle

import torch
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径

pdb_folder = os.path.join(current_dir, "data", "pdb_info")
out_path= os.path.join(current_dir, "data", "norm_factor")

# 确保输出文件夹存在
os.makedirs(out_path, exist_ok=True)

out_path="D:/MachineLearning/蛋白质/蛋白质柔性/大型正式训练/跨膜蛋白_案例分析/data/norm_factor"
def get_norm_Bfactor(pdb_list,chain_id):
    # print(type(pdb_list[0][5]))

    for x in pdb_list:
        x[5] = float(x[5])
    # print(type(pdb_list[0][5]))
    # 将温度因子值提取出来，存放到tensor中
    temp_factors_tensor = torch.tensor([x[5] for x in pdb_list])
    # 计算平均值和标准差
    mean = torch.mean(temp_factors_tensor)
    std = torch.std(temp_factors_tensor)
    mean = float(mean)
    std = float(std)

    # 计算新的温度因子值
    norm_factor = []
    for x in pdb_list:
        temp = []
        if x[3]==chain_id:
            temp_factor = (x[5] - mean) / std
            # print(temp_factor)
            a = float(temp_factor)
            # print(a)
            temp.append(x[2])
            temp.append(x[3])
            temp.append(x[4])
            temp.append(a)
            norm_factor.append(temp)

    return norm_factor
        # print(a)

def two_classes_nonstrict(factor_list):
    factor_labels = []
    for item in factor_list:
        a = item[3]
        if a >= -0.3:  # 非严格分类情况
            factor_labels.append(1)
        else:
            factor_labels.append(0)
    return factor_labels





# 提取B因子列表，存储为单个蛋白质的pkl文件
with open("example.fasta","r") as fasta_file:
    for line in fasta_file:
        if line.startswith(">"):
            protein_id=line[1:6]
            chain_id=protein_id[-1]
            print(protein_id)
            with open(pdb_folder+'/'+protein_id[:-1]+".pickle","rb") as pdb_file:
                pdb_list = pickle.load(pdb_file)
                factor_list = get_norm_Bfactor(pdb_list,chain_id)
                print(factor_list,len(factor_list))
                # labels = two_classes_nonstrict(factor_list)
                with open(out_path+'/'+protein_id+".pickle","wb") as out_file:
                    pickle.dump(factor_list,out_file)

