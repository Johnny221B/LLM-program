import json

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def format_code_files_to_json(file_paths, times, output_file):
    # 确保file_paths和times列表长度一致
    if len(file_paths) != len(times):
        raise ValueError("File paths and times lists must have the same length.")
    
    formatted_data = []
    
    # 遍历文件路径和时间值
    for file_path, time in zip(file_paths, times):
        code = read_file(file_path)
        code = code.strip()
        formatted_data.append({"code": code.strip(), "time": time})
    
    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

# 示例文件路径列表
file_paths = [
    "/home/jingxuan/base_task/basic_task1/train_resnet50.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet509.py",
    "/home/jingxuan/base_task/basic_task2/run_bert.py",
    "/home/jingxuan/base_task/basic_task2/run_bert9.py",
    "/home/jingxuan/base_task/basic_task1/train_gan.py",
    "/home/jingxuan/base_task/basic_task1/train_gan9.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT9.py"
]

# 示例时间值列表
times = [
        6680.702, 9445,
        39173, 45767, 
        916.043, 1572, 
        19304.295, 28168
        ]

# 使用示例
format_code_files_to_json(file_paths, times, 'test_dataset2.json')

file_path3 = [
    "/home/jingxuan/base_task/basic_task1/train_gan_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style2.py"
]

times3=[
    1884, 1860,
    30624, 31180,
    48147, 47899,
    8880, 8838,
]

format_code_files_to_json(file_path3, times3, 'test_dataset3.json')

file_path4 = [
    "/home/jingxuan/base_task/basic_task1/train_gan_style3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan_style4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style4.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style4.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style3.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style4.py"
]

format_code_files_to_json(file_path4, times3, 'test_dataset4.json')
