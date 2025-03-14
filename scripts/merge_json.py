import json

# 读取第一个JSON文件
with open('../data/interview-llm-qa.json', 'r', encoding='utf-8') as f:
    data_lla = json.load(f)

# 读取第二个JSON文件
with open('../data/interview-qa.json', 'r', encoding='utf-8') as f:
    data_qa = json.load(f)

# 合并两个列表
merged_data = data_lla + data_qa

# 将合并后的数据写入新文件
with open('../data/merged_interview_qa.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print("合并完成！保存为 merged_interview_qa.json")