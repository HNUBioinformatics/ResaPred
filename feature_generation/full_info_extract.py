import array

import os
import pickle

import numpy as np
import torch
from Bio import SeqIO
import shutil

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))


# 使用相对路径构建输入输出路径
pdb_folder = os.path.join(current_dir, "data", "pdb_info")
dssp_folder = os.path.join(current_dir, "data", "dssp_info")
pssm_folder = os.path.join(current_dir, "data", "pssm_info")
fasta_file = os.path.join(current_dir, "example.fasta")
feature_folder = os.path.join(current_dir, "data", "full_info")
B_dic = os.path.join(current_dir, "data", "norm_factor")



os.makedirs(feature_folder, exist_ok=True)


# 字典映射onehot,将残基名称当做key可以直接提取相应特征
aa_to_idx = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3,
    'C': 4, 'Q': 5, 'E': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11,
    'M': 12, 'F': 13, 'P': 14, 'S': 15,
    'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

aa_to_name_dic = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                  'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                  'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                  'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                  }



aa_to_maxasa = {
    'A': 115, 'R': 225, 'N': 160, 'D': 150,
    'C': 135, 'Q': 180, 'E': 190, 'G': 75,
    'H': 195, 'I': 175, 'L': 170, 'K': 200,
    'M': 185, 'F': 210, 'P': 145, 'S': 115,
    'T': 140, 'W': 155, 'Y': 230, 'V': 155
}




#
# 'PYL': 'K', 'MSE': 'M', 'CYG': 'C', 'TRN': 'W', 'CSD': 'C', 'CSO': 'C', 'TRO': 'T', 'MME': 'M', 'CSS': 'C', 'SEP': 'S'
# , 'PCA': 'Q', 'LLP': ''
# aa_to_idx = {
#     'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3,
#     'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7,
#     'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11,
#     'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15,
#     'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
# }

# 二级结构类型到编码的映射字典
ss_to_idx = {
    '-': 0, 'H': 1, 'E': 2, 'G': 3,
    'I': 4, 'S': 5, 'T': 6, 'B': 7
}


# 定义函数，提取pdb文件名字的第四位到第七位并转成大写
def get_pdb_id(filename):
    if len(filename)>9:
        return filename[3:7].upper()
    else:
        return filename[:4].upper()

def get_chain(protein_id):
    return protein_id[-1]


# 得到各自的信息列表
def get_pdb_list(protein_id, chain_id):
    pdb_list = []
    pdb_path = pdb_folder + '/' + protein_id[:-1] + ".pickle"
    pdb_file = open(pdb_path, "rb")
    temp = pickle.load(pdb_file)
    pdb_file.close()
    for item in temp:
        if item[3] == chain_id:
            pdb_list.append(item)

    return pdb_list


def get_dssp_list(protein_id, chain_id):
    dssp_list = []
    dssp_path = dssp_folder + '/' + protein_id[:-1] + ".pickle"
    dssp_file = open(dssp_path, "rb")
    temp = pickle.load(dssp_file)
    dssp_file.close()
    for item in temp:
        if item[1] == chain_id:
            dssp_list.append(item)
    return dssp_list


def get_pssm_list(protein_id):
    pssm_path = pssm_folder + '/' + protein_id + ".pickle"
    pssm_file = open(pssm_path, "rb")
    pssm_list = pickle.load(pssm_file)
    pssm_file.close()
    return pssm_list


def get_pdb_dihedral_list(protein_id):  # 只含有标准残基且对应链的列表
    pdb_dihedral_path = pdb_dssp_folder + '/' + protein_id + ".pickle"
    file = open(pdb_dihedral_path, "rb")
    pdb_dihedral_list = pickle.load(file)
    # print(pdb_dihedral_list)
    file.close()
    return pdb_dihedral_list


def onehot_encode(idx, num_classes):
    onehot = torch.zeros(num_classes, dtype=int)
    onehot[idx] = 1
    onehot = onehot.tolist()
    return onehot


def get_aaindex_info(aa_name):
    with open("AAindex_归一化.pickle", "rb") as f:
        aaindex_dic = pickle.load(f)
        physiochemicaldic = aaindex_dic[aa_name]

    return physiochemicaldic


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def get_pdb_seq(pdb_list, chain_id):
    seq = []
    for item in pdb_list:
        if item[3] == chain_id and (item[0] == 'ATOM' and (item[2] in aa_to_name_dic.keys())):
            # print(item[2],chain_id)
            if len(item[2]) == 4:
                aa = aa_to_name_dic[item[2][1:]]
                seq.append(aa)
            else:
                aa = aa_to_name_dic[item[2]]
                seq.append(aa)
    pdb_seq = "".join(seq)
    # print(len(pdb_seq),len(pdb_list))
    return pdb_seq


def find_char_position(pdb_seq, pssm_seq):
    if len(pdb_seq) <= len(pssm_seq):
        shorter_str = pdb_seq
        longer_str = pssm_seq
    else:
        shorter_str = pssm_seq
        longer_str = pdb_seq

    position = longer_str.find(shorter_str[0])
    return position

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    scaled_datalist = scaled_data.tolist()
    return scaled_datalist
def get_acc(pdb_dihedral_list):
    temp=[]
    for item in pdb_dihedral_list:
        temp.append(int(item[8]))
    # print(temp)
    acc_list=min_max_scaling(temp)
    return acc_list
def get_pssm_dihedral(pdb_list, dssp_list, pssm_list, chain_id, sequence, pdb_dihedral_list):
    feature = []
    # offset = get_offset(dssp_list, pssm_list, chain_id)
    offset = get_offset(pdb_list, dssp_list, pssm_list, chain_id, sequence, pdb_dihedral_list)
    index=0
    for item in pdb_dihedral_list:
        temp = []
        # if item[3] == chain_id:
        # index = int(item[4]) - int(offset)-6
        # print(index)
        temp.append(item[4])
        temp.append(item[5])
        temp.append(item[6])
        temp.append(item[7])
        temp.append(item[8])
        temp.append(item[9])
        temp.append(item[10])
        # print(index,item[4],offset)
        feature.append(temp + pssm_list[index][2:])
        index+=1
    # print(feature)
    # print(feature)
    return feature


# def get_offset(dssp_list,pssm_first_aa,chain_id):
#     for item in dssp_list:
#         if item[1] == chain_id:
#             offset = int(item[0]) - int(pssm_first_aa)
#             break
#     return offset

def get_offset(pdb_list, dssp_list, pssm_list, chain_id, sequence, pdb_dihedral_list):
    pdb_seq = get_pdb_seq(pdb_list, chain_id)
    pssm_first_aa = find_char_position(pdb_seq, sequence)
    offset = int(pdb_dihedral_list[0][4]) - int(pssm_list[pssm_first_aa][0])
    # for item in pdb_list:
    #     if item[3] == chain_id:
    #         offset = int(item[0]) -
    #         break
    return offset


def combine_feature(pdb_list, dssp_list, pssm_list, chain_id, sequence, pdb_dihedral_list):
    feature = []
    # print(len(pdb_list))
    pdb_seq = get_pdb_seq(pdb_list, chain_id)
    # print(len(pdb_seq),len(dssp_list))
    # print(len(dssp_list),len(pdb_seq))
    # offset = get_offset(pdb_list,dssp_list, pssm_list,chain_id,sequence)
    aa_pssm_dihedral = get_pssm_dihedral(pdb_list, dssp_list, pssm_list, chain_id, sequence, pdb_dihedral_list)
    count = 0
    acc_list=get_acc(pdb_dihedral_list)
    # print(acc_list)
    for item in aa_pssm_dihedral:
        dihedral = []
        # if item[1]==chain_id:
        # print(dssp)
        # index = int(dssp[0]) - int(offset) - 1
        # temp_list = []
        # print(dssp,pdb_seq[count])
        aa_name = pdb_seq[count]
        # print(dssp[2],aa_name,count)
        aa_onehot = onehot_encode(aa_to_idx[aa_name], 20)
        # print(aa_onehot)
        aa_aaindex = get_aaindex_info(aa_name)
        ss=item[3]
        ss_onehot = onehot_encode(ss_to_idx[ss], 8)

        # temp_list.append(aa_onehot)
        # temp_list.append(aa_aaindex)
        phi_norm = sigmoid(float(item[5]))
        psi_norm = sigmoid(float(item[6]))
        dihedral.append(phi_norm)
        dihedral.append(psi_norm)
        # feature.append(item[:3] + item[5:] + aa_aaindex + aa_onehot + dihedral)
        temp=item[:7]
        # temp.append(acc_list[count])
        feature.append(temp)
        count += 1
        # print(sequence[347])
    return feature




def batch_deal():
    # 读取FASTA文件
    sequences = SeqIO.parse(fasta_file, "fasta")
    feature_dic = {}
    # 遍历每个序列并打印ID和序列
    for seq_record in sequences:
        protein_id = seq_record.id
        print(protein_id)
        sequence = seq_record.seq
        chain_id = protein_id[-1]
        pdb_list = get_pdb_list(protein_id, chain_id)
        dssp_list = get_dssp_list(protein_id, chain_id)
        pssm_list = get_pssm_list(protein_id)
        pdb_dihedral_list = get_pdb_dihedral_list(protein_id)
        feature = combine_feature(pdb_list, dssp_list, pssm_list, chain_id, sequence, pdb_dihedral_list)
        print(len(feature[0]),feature)
        feature_dic[protein_id] = feature
        print(protein_id, len(feature_dic[protein_id]))
        with open(feature_folder + '/' + protein_id + ".pickle", "wb") as file:
            pickle.dump(feature, file)

    # with open("features.pickle", "wb") as f:
    #     pickle.dump(feature_dic, f)


# single_deal()
batch_deal()















































































