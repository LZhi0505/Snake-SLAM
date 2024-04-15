#!/bin/bash

# 获取脚本所在的绝对路径
script_dir=$(dirname "$(readlink -f "$0")")
echo "script_dir: $script_dir"

# 切换到主目录
cd "$script_dir"

# 获取Git记录中被修改的文件列表（不包含"3rd"目录）
modified_files=$(git status --porcelain | awk '{print $2}' | grep -v "build" | grep -v "cmake" | grep -v "cmake-build-debug" | grep -v "configs" | grep -v "External")

# 支持的C++文件类型
cpp_types=("h" "c" "hpp" "cc" "cpp" "tpp")

# 遍历被修改的文件并使用clang-format进行格式化
for src_file in $modified_files
do
    # 获取文件扩展名
    extension="${src_file##*.}"
    # 检查文件扩展名是否在支持的类型列表中
    for value in "${cpp_types[@]}"
    do
        if [[ "$value" == "$extension" ]];then
          echo "clang-format $src_file format $extension"
          clang-format --assume-filename=.clang-format -i "$src_file"
        fi
    done
done
