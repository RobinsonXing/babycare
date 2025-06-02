import os

sequence_dir = './data_aug/ChatGPT-o4-instructed/sequence'
label_dir = './data_aug/ChatGPT-o4-instructed/label'


os.makedirs(sequence_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

def find_next_empty_file_pair(max_index=10000):
    for i in range(max_index):
        file_id = f"A{i:05d}"
        seq_path = os.path.join(sequence_dir, f"{file_id}.csv")
        label_path = os.path.join(label_dir, f"{file_id}_label.csv")
        if os.path.exists(seq_path) and os.path.exists(label_path):
            if os.path.getsize(seq_path) == 0 and os.path.getsize(label_path) == 0:
                return seq_path, label_path
    return None, None

print("[Log] Start to write data.")

while True:
    print("\n[Log] write sequence data including header (or enter 'exit' to terminate)：")
    seq_lines = []
    while True:
        line = input()
        if line.strip().lower() == "exit":
            print("[log] exited.")
            exit(0)
        if line.strip() == "":
            break
        seq_lines.append(line)

    print("[Log] write label data including header (or enter 'exit' to terminate)")
    label_lines = []
    while True:
        line = input()
        if line.strip().lower() == "exit":
            print("[log] exited.")
            exit(0)
        if line.strip() == "":
            break
        label_lines.append(line)

    seq_file, label_file = find_next_empty_file_pair()

    if not seq_file or not label_file:
        print("[Log] Automatically terminate for no blank file。")
        break

    with open(seq_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(seq_lines) + '\n')

    with open(label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_lines) + '\n')

    print(f"[Log] have been written to：\n - {seq_file}\n - {label_file}")