import pickle
import os

import numpy as np
import torch
from Bio import SeqIO

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))


# 使用相对路径构建输入输出路径
pdb_folder = os.path.join(current_dir, "data", "pdb_info")
dssp_folder = os.path.join(current_dir, "data", "dssp_info")
pssm_folder = os.path.join(current_dir, "data", "pssm_info")
fasta_file = os.path.join(current_dir, "example.fasta")
pdb_dssp_folder = os.path.join(current_dir, "data", "pdb_dssp_info")

os.makedirs(pdb_dssp_folder, exist_ok=True)


aa_to_name_dic={'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
def get_pdb_list(protein_id,chain_id):
    pdb_list=[]
    pdb_path=pdb_folder+'/'+protein_id[:-1]+".pickle"
    pdb_file=open(pdb_path,"rb")
    temp=pickle.load(pdb_file)
    pdb_file.close()
    for item in temp:
        if item[3] == chain_id:
            pdb_list.append(item)

    return pdb_list
def get_dssp_list(protein_id,chain_id):
    dssp_list=[]
    dssp_path=dssp_folder+'/'+protein_id[:-1]+".pickle"
    dssp_file=open(dssp_path,"rb")
    temp=pickle.load(dssp_file)
    dssp_file.close()
    for item in temp:
        if item[1]==chain_id:
            dssp_list.append(item)
    return dssp_list
def get_pdb_dihedral(pdb_list,dssp_list):
    pdb_dssp_list=[]
    pdb_dihedral=[]
    for i in range(len(pdb_list)):
        temp = []
        temp.append(pdb_list[i][0])
        temp.append(pdb_list[i][1])
        temp.append(pdb_list[i][2])
        temp.append(pdb_list[i][3])
        temp.append(pdb_list[i][4])
        temp.append(pdb_list[i][5])
        temp.append(dssp_list[i][2])
        temp.append(dssp_list[i][3])
        temp.append(dssp_list[i][4])
        temp.append(dssp_list[i][5])
        temp.append(dssp_list[i][6])
        pdb_dssp_list.append(temp)
    for item in pdb_dssp_list:
        if item[0]=='ATOM' and (item[2] in aa_to_name_dic):
            pdb_dihedral.append(item)
    return pdb_dihedral

sequences = SeqIO.parse(fasta_file, "fasta")
feature_dic = {}
# 遍历每个序列并打印ID和序列
for seq_record in sequences:
    protein_id = seq_record.id
    print(protein_id)
    sequence = seq_record.seq
    chain_id=protein_id[-1]
    pdb_list=get_pdb_list(protein_id,chain_id)
    dssp_list = get_dssp_list(protein_id,chain_id)
    pdb_dihedral=get_pdb_dihedral(pdb_list,dssp_list)
    with open(pdb_dssp_folder+'/'+protein_id+".pickle","wb") as f:
        pickle.dump(pdb_dihedral,f)
