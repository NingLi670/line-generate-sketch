import os

folder_path = '/home/huteng/zhuhaokun/4-i/CGI-PSG-Training_Set/style/train_sketch' # 将 'your_folder_path' 替换为您想要重命名文件的文件夹路径
new_file_name = 'new_file_name'  # 将 'new_file_name' 替换为您想要给文件命名的新名称
counter = 420


files = os.listdir(folder_path)
files.sort(key=lambda x:-int(x[:-4]))
for filename in files:
    file_path = os.path.join(folder_path, filename)
    print(file_path)
    if os.path.isfile(file_path):
        file_extension = os.path.splitext(filename)[1] # 获取文件扩展名
        new_filename = str(counter+180) + file_extension # 组成新文件名
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(file_path, new_file_path) # 重命名文件
        counter -= 1
