import os

# 指定存放加速度和标签的目录
sequence_dir = '/fast/workspace/robinson/CodeSource/babycare/data_aug/ChatGPT-o4-instructed/sequence'
label_dir = '/fast/workspace/robinson/CodeSource/babycare/data_aug/ChatGPT-o4-instructed/label'

# 自动找到第一个空文件对
def find_next_empty_file_pair(max_index=10000):
    for i in range(max_index):
        file_id = f"A{i:05d}"
        seq_path = os.path.join(sequence_dir, f"{file_id}.csv")
        label_path = os.path.join(label_dir, f"{file_id}_label.csv")
        if os.path.exists(seq_path) and os.path.exists(label_path):
            if os.path.getsize(seq_path) == 0 and os.path.getsize(label_path) == 0:
                return seq_path, label_path
    return None, None

print("🟢 开始循环写入数据，输入 'exit' 后按回车 可退出。")

while True:
    print("\n请粘贴第一段加速度数据（包含表头），输入 'exit' 退出：")
    seq_lines = []
    while True:
        line = input()
        if line.strip().lower() == "exit":
            print("⛔️ 已退出。")
            exit(0)
        if line.strip() == "":
            break
        seq_lines.append(line)

    print("请粘贴第二段标签数据（包含表头），输入 'exit' 退出：")
    label_lines = []
    while True:
        line = input()
        if line.strip().lower() == "exit":
            print("⛔️ 已退出。")
            exit(0)
        if line.strip() == "":
            break
        label_lines.append(line)

    seq_file, label_file = find_next_empty_file_pair()

    if not seq_file or not label_file:
        print("❌ 未找到空文件对。已终止。")
        break

    with open(seq_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(seq_lines) + '\n')

    with open(label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_lines) + '\n')

    print(f"✅ 已写入到：\n - {seq_file}\n - {label_file}")