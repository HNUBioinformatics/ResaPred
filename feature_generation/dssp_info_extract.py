# 创建PDB文件解析器对象,提取dssp的二级结构
import os
import pickle

from Bio import SeqIO


# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径
dssp_folder = os.path.join(current_dir, "dssp")
output_folder = os.path.join(current_dir, "data", "dssp_info")

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)




# 定义函数，提取pdb文件名字的第四位到第七位并转成大写
def get_dssp_id(filename):
    if len(filename)>9:
        return filename[3:7].upper()
    else:
        return filename[:4].upper()

def get_dssp_info(dssp_file):
    with open(dssp_file,"r") as f:
        dssp_info_list=[]
        data = f.readlines()[28:]
        for line in data:
            reds_list = []
            reds_list.append(line[5:10].replace(" ",""))
            reds_list.append(line[10:12].replace(" ",""))
            reds_list.append(line[12:14].replace(" ",""))
            reds_list.append(line[16:17].replace(" ","-"))
            reds_list.append(line[34:38].replace(" ",""))
            reds_list.append(line[103:109].replace(" ",""))
            reds_list.append(line[109:115].replace(" ",""))
            # print(reds_list)
            dssp_info_list.append(reds_list)
        return dssp_info_list




# 读取fasta文件中的蛋白质ID
with open("example.fasta", "r") as f:
    protein_ids = set(record.id[:4] for record in SeqIO.parse(f, "fasta"))
    print(protein_ids)

# 遍历包含pdb文件的文件夹

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in os.listdir(dssp_folder):
    # 如果文件名以".pdb"结尾，则提取pdb文件名字的第四位到第七位并转成大写
    if filename.endswith(".dssp"):
        dssp_id = get_dssp_id(filename)
        print(dssp_id)
        # 如果pdb文件名与fasta文件中的蛋白质ID相同，则将该pdb文件复制到新文件夹中
        if dssp_id in protein_ids:
            print(filename)
            dssp_info_list=get_dssp_info(dssp_folder+'/'+filename)
            # print(dssp_info_list)
            with open(output_folder+'/'+dssp_id+'.pickle',"wb") as f:
                pickle.dump(dssp_info_list,f)
