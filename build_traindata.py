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
        formatted_data.append({"code": code.strip(), "time": time})
    
    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

# 示例文件路径列表
file_paths3 = [
    "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet505.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet506.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet507.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet508.py",
    "/home/jingxuan/base_task/basic_task2/run_bert1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert4.py",
    "/home/jingxuan/base_task/basic_task2/run_bert5.py",
    "/home/jingxuan/base_task/basic_task2/run_bert6.py",
    "/home/jingxuan/base_task/basic_task2/run_bert7.py",
    "/home/jingxuan/base_task/basic_task2/run_bert8.py",
    "/home/jingxuan/base_task/basic_task1/train_gan1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan2.py",
    "/home/jingxuan/base_task/basic_task1/train_gan3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan4.py",
    "/home/jingxuan/base_task/basic_task1/train_gan5.py",
    "/home/jingxuan/base_task/basic_task1/train_gan6.py",
    "/home/jingxuan/base_task/basic_task1/train_gan7.py",
    "/home/jingxuan/base_task/basic_task1/train_gan8.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT5.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT6.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT7.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT8.py"
]

times3 = [
        7541.934, 12565.417, 8836.583, 10061.487, 19157, 17874, 11458, 15092,
        19234, 22309.981, 28862.42, 51294.6, 64717.769, 48919.793, 70738.926, 42498.494,
        1839.131, 548.525, 1200.722, 1653.865, 2761, 2233, 2424, 2104,
        23053.283, 37774.016, 31033.167, 21068.562, 42152, 45572, 16926, 33760
        ]

format_code_files_to_json(file_paths3, times3, 'train_dataset3.json')

file_paths = [
    "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
    "/home/jingxuan/base_task/basic_task2/run_bert1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert4.py",
    "/home/jingxuan/base_task/basic_task1/train_gan1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan2.py",
    "/home/jingxuan/base_task/basic_task1/train_gan3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT4.py"
]

times = [
        7541.934, 12565.417, 8836.583, 10061.487,
        19234, 22309.981, 28862.42, 51294.6, 
        1839.131, 548.525, 1200.722, 1653.865,
        23053.283, 37774.016, 31033.167, 21068.562
        ]
format_code_files_to_json(file_paths, times, 'train_dataset.json')

file_paths2 = [
    "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
    "/home/jingxuan/base_task/basic_task2/run_bert1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert4.py",
    "/home/jingxuan/base_task/basic_task2/run_bert5.py",
    "/home/jingxuan/base_task/basic_task2/run_bert6.py",
    "/home/jingxuan/base_task/basic_task2/run_bert7.py",
    "/home/jingxuan/base_task/basic_task2/run_bert8.py",
    "/home/jingxuan/base_task/basic_task1/train_gan1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan2.py",
    "/home/jingxuan/base_task/basic_task1/train_gan3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT4.py"
]

times2 = [
        7541.934, 12565.417, 8836.583, 10061.487,
        19234, 22309.981, 28862.42, 51294.6, 64717.769, 48919.793, 70738.926, 42498.494,
        1839.131, 548.525, 1200.722, 1653.865,
        23053.283, 37774.016, 31033.167, 21068.562
        ]
format_code_files_to_json(file_paths2, times2, 'train_dataset2.json')