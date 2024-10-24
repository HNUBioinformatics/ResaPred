import pickle
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径
factor_path = os.path.join(current_dir, "data", "norm_factor")
features_path = os.path.join(current_dir, "data", "full_info")
output_path = os.path.join(current_dir, "data", "Bfactor_labels")

# 确保输出文件夹存在
os.makedirs(output_path, exist_ok=True)


def get_feature_list(protein_id):
    path=features_path+'/'+protein_id+".pickle"
    with open(path,"rb") as f:
        feature_list=pickle.load(f)
    return feature_list

def get_norm_factor_list(protein_id):
    path=factor_path+'/'+protein_id+".pickle"
    with open(path,"rb") as f:
        norm_factor_list=pickle.load(f)
    return norm_factor_list


def get_bfactor_list(feature_list,norm_factor_list):
    bfactor=[]
    for item in feature_list:
        temp=[]
        for factor in norm_factor_list:
            # print(item,factor)
            if item[0]==factor[2]:
                temp.append(item[0])
                temp.append(item[2])
                temp.append(factor[3])
                bfactor.append(temp)
                break
    return bfactor



#提取B因子，存储为整个字典形式
with open("example.fasta","r") as fasta_file:
    for line in fasta_file:
        if line.startswith(">"):
            protein_id=line[1:6]
            feature_list=get_feature_list(protein_id)
            norm_factor_list = get_norm_factor_list(protein_id)
            bfactor_list = get_bfactor_list(feature_list, norm_factor_list)
            with open(output_path+'/'+protein_id+".pickle","wb") as f:
                pickle.dump(bfactor_list,f)
            print(protein_id,len(norm_factor_list),len(feature_list),len(bfactor_list))
