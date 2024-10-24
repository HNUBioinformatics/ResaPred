#好像是真实数据和假数据分别放入判别器进行分类，将真实数据正确分类和假数据正确被判别的结果一起进行loss函数计算
import pickle
# from plistlib import Data
import torch.optim as optim
import numpy as np
from Bio import SeqIO
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
# from CBAM import *

import torch.utils.model_zoo as model_zoo
import os
import sys

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径
bfactor_path = os.path.join(current_dir, "data", "Bfactor_labels")
features_path = os.path.join(current_dir, "data", "features")
output_path = os.path.join(current_dir, "data", "win9_2d")

# 确保输出文件夹存在
os.makedirs(output_path, exist_ok=True)



#数据导入处理
def appendzero(my_list,windowsize,feature_length):
    new_list=[]
    number=windowsize//2
    for i in range(len(my_list)):
        new_element = []
        for j in range(i - number, i + number+1):
            if j < 0 or j >= len(my_list):
                new_element.append([0]*feature_length)
            else:
                new_element.append(my_list[j])
        new_list.append(new_element)

    # print(new_list)
    return new_list


def pssm_cut(feature_list):
    new_feature_list=[]
    for item in feature_list:
        new_item = item[:8] + item[28:]
        new_feature_list.append(new_item)
    return new_feature_list

def aaindex_cut(feature_list):
    new_feature_list=[]
    for item in feature_list:
        new_item = item[:8] + item[28:]
        new_feature_list.append(new_item)
    return new_feature_list


def get_feature_list(protein_id):
    path=features_path+'/'+protein_id+".pickle"
    with open(path,"rb") as f:
        feature_list=pickle.load(f)
    feature_list=appendzero(feature_list,windowsize=9,feature_length=109)
    return feature_list
def get_bfactor_list(protein_id):
    path=bfactor_path+'/'+protein_id+".pickle"
    with open(path,"rb") as f:
        bfactor_list=pickle.load(f)
    return bfactor_list

#二分类柔性标准
def two_classes_nonstrict(factor_list):
    factor_labels = []
    for item in factor_list:
        a = item[2]
        if a >= -0.3:  # 非严格分类情况
            factor_labels.append(1)
        else:
            factor_labels.append(0)
    # factor_labels = torch.tensor(factor_labels,dtype=int)
    return factor_labels

def two_classes_strict(factor_list):
    factor_labels = []
    for item in factor_list:
        a = item[2]
        if a >= 0.03:  # 严格分类情况
            factor_labels.append(1)
        else:
            factor_labels.append(0)
    # factor_labels=torch.tensor(factor_labels,dtype=int)
    return factor_labels



def concat_all_protein(fasta_file,bfactor_classes):
    with open(fasta_file, "r") as f:
        protein_ids = set(record.id[:5] for record in SeqIO.parse(f, "fasta"))
        count = 0
        all_protein_feature = []
        all_protein_bfactor = []
        for protein_id in protein_ids:
            count += 1
            print(count)
            feature_list = get_feature_list(protein_id)
            bfactor_list = get_bfactor_list(protein_id)
            if bfactor_classes == "strict_two":
                bfactor_list = two_classes_strict(bfactor_list)
                all_protein_feature = all_protein_feature + feature_list
                all_protein_bfactor = all_protein_bfactor + bfactor_list
                with open(output_path + "/strict/" + protein_id + "_strict_features.pickle", "wb") as file1:
                    pickle.dump(all_protein_feature, file1)
                with open(output_path + "/strict/" + protein_id + "_strict_bfactor.pickle", "wb") as file2:
                    pickle.dump(bfactor_list, file2)
                # print(len(feature_list), len(bfactor_list), len(all_protein_feature), len(all_protein_bfactor))

            if bfactor_classes == "nonstrict_two":
                bfactor_list = two_classes_nonstrict(bfactor_list)
                all_protein_feature = all_protein_feature + feature_list
                all_protein_bfactor = all_protein_bfactor + bfactor_list
                with open(output_path + "/nonstrict/" + protein_id + "_nonstrict_features.pickle", "wb") as file1:
                    pickle.dump(all_protein_feature, file1)
                with open(output_path + "/nonstrict/" + protein_id + "_nonstrict_bfactor.pickle", "wb") as file2:
                    pickle.dump(bfactor_list, file2)
                # print(len(feature_list), len(bfactor_list), len(all_protein_feature), len(all_protein_bfactor))
                # print(len(features),len(factor_labels))



#提取B因子，存储为整个字典形式
with open("example.fasta","r") as fasta_file:
    for line in fasta_file:
        if line.startswith(">"):
            protein_id=line[1:6]
            print(protein_id)
            concat_all_protein(protein_id + ".fasta", "nonstrict_two")
            concat_all_protein(protein_id+".fasta", "strict_two")

