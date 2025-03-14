import json
import requests
import time
from typing import List, Dict

# 配置参数
DEEPSEEK_API_KEY = "sk-xxx"  # 替换为你的API密钥
API_URL = "https://api.deepseek.com/v1/chat/completions"  # 确认最新的API地址
MODEL_NAME = "deepseek-chat"  # 根据实际模型名称调整
HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}


def load_existing_questions(file_path: str) -> List[str]:
    """加载现有问题用于去重"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item["input"].strip().lower() for item in data}


def generate_variant_question(original_question: str) -> str:
    """使用DeepSeek生成语义相近的新问题"""
    prompt = f"""请根据以下问题生成一个全新的面试问题，要求：
1. 保持相同的专业领域和技术方向
2. 使用完全不同的表述方式
3. 不包含原问题中的关键词
4. 保持相同的抽象层次
5. 适合作为资深算法工程师的面试题
6. 需要结合实际，有应用场景和需求
原问题：{original_question}
新问题："""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"生成问题时发生错误: {str(e)}")
        return ""


def generate_answer(question: str) -> str:
    """生成对应问题的专业答案"""
    prompt = f"""请以专业面试者的角度，用中文简要回答以下算法工程师面试问题，回答时不需要重复问题：
{question}
答案："""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"生成答案时发生错误: {str(e)}")
        return ""


def expand_dataset(input_file: str, output_file: str, expansion_factor: int = 2):
    """扩展数据集核心函数"""

    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 创建问题集合用于去重
    existing_questions = load_existing_questions(input_file)
    new_entries = []

    # 遍历每个原始问题生成变体
    for item in original_data:
        original_question = item["input"]
        print(f"正在处理原始问题: {original_question}")

        for _ in range(expansion_factor):
            # 生成新问题
            new_question = generate_variant_question(original_question)

            # 有效性检查
            if not new_question:
                continue

            # 去重检查
            normalized = new_question.strip().lower()
            if normalized in existing_questions:
                print(f"跳过重复问题: {new_question}")
                continue

            # 生成答案
            new_answer = generate_answer(new_question)

            if new_answer:
                # 构建新条目
                new_entry = {
                    "instruction": item["instruction"],
                    "input": new_question,
                    "output": new_answer
                }
                new_entries.append(new_entry)
                existing_questions.add(normalized)
                print(f"成功生成新条目: {new_entry}")
                exit(1)


            # 避免速率限制
            time.sleep(1)

    # 合并新旧数据
    combined_data = original_data + new_entries

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"生成完成！共新增 {len(new_entries)} 条数据，总数据量 {len(combined_data)} 条")


# 使用示例
if __name__ == "__main__":
    expand_dataset(
        input_file="../data/merged_interview_qa.json",  # 输入文件路径
        output_file="expanded_qa_dataset.json",  # 输出文件路径
        expansion_factor=2  # 每个问题生成2个变体
    )