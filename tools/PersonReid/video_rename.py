import os

# 设置目录路径
directory = 'dataset/person/gallery/'

# 获取目录下的所有文件
files = os.listdir(directory)

# 遍历文件列表
for index, filename in enumerate(files):
    # 检查是否是视频文件，这里只是简单的检查扩展名为.mp4
    if filename.endswith('.mp4'):
        # 获取城市名
        city = filename.split('.')[0]
        # 构建新的文件名
        new_filename = f'PersonReid_{city}.mp4'
        # 生成旧文件路径和新文件路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')

