import os
import pickle
from Bio import SeqIO
import shutil

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建输入输出路径
pdb_folder = os.path.join(current_dir, "pdb")
output_folder = os.path.join(current_dir, "data", "pdb_info")

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# def get_Bfactor(pdb_file):
def get_pdb_info(pdb_file):
    with open(pdb_file, "r") as f:
        reds_info = []
        for line in f:
            line_list = []
            if (line.startswith("ATOM") and line[13:16].replace(" ", "") == "CA") or (
                    line.startswith("HETATM") and line[13:16].replace(" ", "") == "CA"):
                if len(line[16:21].replace(" ", ""))==4:
                    if line[16:21].replace(" ", "")[0]=='A':
                        line_list.append(line[:6].replace(" ", ""))
                        line_list.append(line[13:16].replace(" ", ""))
                        line_list.append(line[16:21].replace(" ", ""))
                        line_list.append(line[21:22].replace(" ", ""))
                        line_list.append(line[22:27].replace(" ", ""))
                        line_list.append(line[60:67].replace(" ", ""))
                    else:
                        continue
                else:
                    line_list.append(line[:6].replace(" ", ""))
                    line_list.append(line[13:16].replace(" ", ""))
                    line_list.append(line[16:21].replace(" ", ""))
                    line_list.append(line[21:22].replace(" ", ""))
                    line_list.append(line[22:27].replace(" ", ""))
                    line_list.append(line[60:67].replace(" ", ""))
                # print(len(line),line[:6],line[6:11],line[13:17],line[17:21],line[21:23],line[23:27],line[61:67])

                reds_info.append(line_list)
        return reds_info
    # for i in range(len(reds_info)):
    #     print(reds_info[i])


# 定义函数，提取pdb文件名字的第四位到第七位并转成大写
def get_pdb_id(filename):
    if len(filename)>9:
        return filename[3:7].upper()
    else:
        return filename[:4].upper()

# 读取fasta文件中的蛋白质ID
with open("example.fasta", "r") as f:
    protein_ids = set(record.id[:4] for record in SeqIO.parse(f, "fasta"))
    print(protein_ids)

# 遍历包含pdb文件的文件夹

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in os.listdir(pdb_folder):
    # 如果文件名以".pdb"结尾，则提取pdb文件名字的第四位到第七位并转成大写
    if filename.endswith(".pdb"):
        pdb_id = get_pdb_id(filename)
        # print(pdb_id)
        # 如果pdb文件名与fasta文件中的蛋白质ID相同，则将该pdb文件复制到新文件夹中
        if pdb_id in protein_ids:
            print(filename)
            pdb_info_list=get_pdb_info(pdb_folder+'/'+filename)
            # print(pdb_info_list)
            with open(output_folder+'/'+pdb_id+'.pickle',"wb") as f:
                pickle.dump(pdb_info_list,f)
































# with open("1a62.pkl","wb") as f:
#     pickle.dump(reds_info,f)

# with open("1a62.pkl","rb") as f:
#     data=pickle.load(f)
#     print(data)

