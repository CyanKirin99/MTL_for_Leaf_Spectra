import glob
from sklearn.model_selection import train_test_split

# 获取所有.pkl文件的路径
files = glob.glob('use_pkl/*.pkl')

# 使用某些数据集
files = [f for f in files if '\\90' not in f]

# 按照70%、15%、15%的比例划分训练集、验证集和测试集
train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# 将划分结果保存为txt文件
with open('train_files.txt', 'w') as f:
    for file in train_files:
        f.write("data/%s\n" % file)

with open('val_files.txt', 'w') as f:
    for file in val_files:
        f.write("data/%s\n" % file)

with open('test_files.txt', 'w') as f:
    for file in test_files:
        f.write("data/%s\n" % file)
