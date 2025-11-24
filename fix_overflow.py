#!/usr/bin/env python3
"""
修复 MCTS.py 中的整数溢出问题
将 self.score 从隐式int8改为显式int类型
"""

import re

file_path = "mcts/MCTS.py"

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 备份原文件
with open(file_path + '.backup', 'w', encoding='utf-8') as f:
    f.write(content)

# 修复1: 初始化时明确使用int类型
content = content.replace(
    '        self.score = 0',
    '        self.score = int(0)  # Use Python int to avoid overflow'
)

# 修复2: 在copy方法中也使用int
content = content.replace(
    '        new_state.score = self.score',
    '        new_state.score = int(self.score)'
)

# 修复3: 确保_process_merges中的加法使用int
# 找到 self.score += reward 这一行并修改
pattern = r'(\s+)(self\.score \+= reward)'
replacement = r'\1self.score = int(self.score) + int(reward)  # Prevent overflow'
content = re.sub(pattern, replacement, content)

# 写入修复后的文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 修复完成！")
print(f"✅ 原文件已备份为: {file_path}.backup")
print("\n修复内容：")
print("1. self.score 初始化时使用 int(0)")
print("2. copy 方法中使用 int(self.score)")
print("3. score 累加时使用 int 转换防止溢出")
