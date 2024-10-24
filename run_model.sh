#!/bin/bash

# 设置固定路径
# 获取当前脚本所在目录
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# 设置固定路径
BASE_PATH="$SCRIPT_DIR"

# 设置输入输出路径，dssp,pdb,pssm这三个文件夹内需要提前放入自行准备的对应的蛋白质文件，此外，请将蛋白质的fasta序列放入example.fasta文件中
dssp_folder="$BASE_PATH/dssp"
pdb_folder="$BASE_PATH/pdb"
pssm_folder="$BASE_PATH/pssm"
fasta_file="$BASE_PATH/example.fasta"
data_folder="$BASE_PATH/data"
mode_para_path="$BASE_PATH/model_parameter"

# 创建输出文件夹，如果文件夹创建失败，可跳过该步骤，运行Python文件，Python文件会自行创建
mkdir -p "$data_folder/dssp_info"
mkdir -p "$data_folder/pdb_info"
mkdir -p "$data_folder/pssm_info"
mkdir -p "$data_folder/pdb_dssp_info"
mkdir -p "$data_folder/norm_factor"
mkdir -p "$data_folder/full_info"
mkdir -p "$data_folder/features"
mkdir -p "$data_folder/Bfactor_labels"
mkdir -p "$data_folder/win9_2d"


# 执行Python文件,如果运行失败，可按照下列顺序手动运行
python3 "$BASE_PATH/feature_generation/dssp_info_extract.py"
python3 "$BASE_PATH/feature_generation/pdb_info_extract.py"
python3 "$BASE_PATH/feature_generation/pssm_info_extract.py"
python3 "$BASE_PATH/feature_generation/pdb_dssp_align.py"
python3 "$BASE_PATH/feature_generation/Bfactor_extract.py"
python3 "$BASE_PATH/feature_generation/full_info_extract.py"
python3 "$BASE_PATH/feature_generation/feature_info_extract.py"
python3 "$BASE_PATH/feature_generation/label_generate.py"
python3 "$BASE_PATH/feature_generation/win9feat_generation.py"
python3 "$BASE_PATH/model.py"

echo "所有文件已成功执行。"
