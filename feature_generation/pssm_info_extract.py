# 设置文件路径
import os
import pickle
import numpy as np

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径
dssp_folder = os.path.join(current_dir, "pssm")
outpath = os.path.join(current_dir, "data", "pssm_info")

# 确保输出文件夹存在
os.makedirs(outpath, exist_ok=True)


def get_pssm_info(pssm_file):
    # 打开文件并跳过前三行
    with open(pssm_file, "r") as f:
        for _ in range(3):
            next(f)
        # 读取文件内容并将pssm进化信息存储到列表中
        pssm_info = []
        each_reds_pssm = []
        count=1
        for line in f:
            each_reds_pssm = []
            if (line.isspace()) == False:
                line = line.strip("\t")
                strline = str(line)
                list_line = strline.split()
                list_line[0]=count
                count+=1
                # print(list_line[1])
                for i in range(22):
                    each_reds_pssm.append(list_line[i])
                # print(each_reds_pssm,len(each_reds_pssm))
                pssm_info.append(each_reds_pssm)
            if line.isspace():
                break
        return pssm_info
    # print(len(pssm_info))

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig
def pssm_norm(pssm_info):#归一化，参考结合位点那篇博士论文
    for i in range(len(pssm_info)):
        for j in range(20):
            pssm_info[i][j+2]=sigmoid(int(pssm_info[i][j+2]))
    return pssm_info

# pssm_info=get_pssm_info(pssm_file)
# # print(pssm_info)
# pssm_info_norm=pssm_norm(pssm_info)
# # print(pssm_info_norm)
# strline="1A62A.pssm"
# print(strline[:-5])
# # with open('pssm_test.pickle',"wb") as f:
# #     pickle.dump(pssm_info_norm,f)
# with open("pssm_test.pickle","rb") as file:
#     data=pickle.load(file)


for filename in os.listdir(pssm_folder):
    if filename.endswith(".pssm"):
        print(filename[:-5],filename)
        pssm_file=pssm_folder+'/'+filename
        pssm_info=get_pssm_info(pssm_file)
        pssm_info_norm=pssm_norm(pssm_info)
        with open(outpath+'/'+filename[:-5]+'.pickle',"wb") as f:
            pickle.dump(pssm_info_norm,f)



# with open(outpath+'/'+"8PN6A.pickle","rb") as pdb_file:
#     pdb_list=pickle.load(pdb_file)
#     # factor_list=get_norm_Bfactor(pdb_list,'A')
#     # for i in pdb_list:
#     #     print(i)
#     print(len(pdb_list))
