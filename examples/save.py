import numpy as np

# 1️⃣ 创建一些数组
arr1 = np.arange(10, dtype=np.int32)            # 0~9 的一维数组
arr2 = np.ones((3, 3), dtype=np.int32)            # 3x3 全 1
arr3 = np.random.rand(2, 4)     # 2x4 随机浮点数数组

# 2️⃣ 保存为 .npz 文件
np.savez("example.npz", a=arr1, b=arr2, c=arr3)
# 如果希望压缩文件，可以用：
# np.savez_compressed("example_compressed.npz", a=arr1, b=arr2, c=arr3)

# 3️⃣ 加载 .npz 文件
data = np.load("example.npz")

print("数组名列表：", data.files)

# 访问数组
arr1_loaded = data['a']
arr2_loaded = data['b']
arr3_loaded = data['c']

print("arr1:", arr1_loaded)
print("arr2:", arr2_loaded)
print("arr3:", arr3_loaded)

# 4️⃣ 可选：转换成普通 dict
arrays_dict = {name: data[name] for name in data.files}
