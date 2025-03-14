import os
import re
import json
import time
import requests
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

# 配置参数
MARKDOWN_PATH = os.path.expanduser("../machine-learning-interview.md")
OUTPUT_JSON = "interview-llm-qa.json"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = "sk-bf951017394346f5b791d662d28c25d6"  # 替换为你的API Key
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

INSTRUCTION = "请以专业面试官的角度，用中文详细回答以下深度学习面试问题："


def split_question_number(full_question: str) -> tuple:
    """分离编号和问题内容"""
    number_pattern = r'^(\d+(-\d+)*)\s+(.*)'
    match = re.match(number_pattern, full_question)
    if match:
        return match.group(1), match.group(3)
    return None, full_question


def extract_questions(md_path: str) -> List[str]:
    question_pattern = re.compile(r'-\s\[(.*?)\]\s\[(.*?)\]\(#.*?\)')
    questions = []
    seen = set()

    with open(md_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = question_pattern.search(line)
            if match:
                # 直接获取完整的问题编号+内容
                full_question = match.group(2).strip()
                question_id, clean_question = split_question_number(full_question)
                # 去除编号
                # clean_question = re.sub(r'^\d+(-\d+)*\s+', '', full_question)
                if question_id and clean_question and clean_question not in seen:
                    seen.add(clean_question)
                    questions.append((question_id, clean_question))
    return questions


def get_parent_group(qid: str) -> str:
    """
    提取编号的父级组别
    :param qid: 问题编号字符串，如 "2-2-1"
    :return: 父级组别字符串，如 "2-2"
    """
    parts = qid.split('-')
    if len(parts) <= 1:
        return qid  # 处理没有分隔符的情况
    return '-'.join(parts[:-1])


def group_questions(questions: list) -> dict:
    """
    问题分组主函数
    :param questions: 问题列表，格式 [(qid, question), ...]
    :return: 分组字典 {父级组别: [问题列表]}
    """
    groups = defaultdict(list)

    for qid, question in questions:
        # 提取父级组别
        parent = get_parent_group(qid)
        # 添加到对应分组
        groups[parent].append((qid, question))

    return dict(groups)


def batch_get_answers(questions: List[str], batch_size=10, max_retries=3) -> List[str]:
    """批量获取回答"""
    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    all_answers = []

    for batch in batches:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": f"请以专业面试者的角度,依次简短地回答以下深度学习面试问题，所有答案以json数组给出,key统一用\"answer\"标识,不需要给出问题:\n" + "\n".join(
                    [f"{i + 1}. {q}" for i, q in enumerate(batch)]
                )}
            ],
            "temperature": 0.3,
            "max_tokens": min(8192, 2048 * batch_size)  # 按需调整
        }

        try:
            response = requests.post(DEEPSEEK_API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            answers = parse_batch_response(response.json(), batch_size)
            print(answers)
            all_answers.extend(answers)
            if answers:
                continue
            else:
                print(f"response:{response}")
        except Exception as e:
            print(f"Attempt failed: {str(e)}, payload:{payload}")

        # response = requests.post(DEEPSEEK_API_URL, headers=HEADERS, json=payload)
        # answers = parse_batch_response(response.json(), batch_size)
    return all_answers



def print_group_queation_tree(group_questions: dict):
    # 可视化输出
    print("{:<8} {:<10} {}".format("组别", "问题编号", "问题摘要"))
    print("-" * 50)
    for parent, items in group_questions.items():
        group_header = f"【{parent}】" if parent.count('-') == 0 else f"  ↳ {parent}"
        for i, (qid, question) in enumerate(items):
            prefix = "├─" if i < len(items) - 1 else "└─"
            print(f"{group_header if i == 0 else '':<8} {prefix} {qid:<10} {question[:20]}...")


def parse_batch_response(response: dict, batch_size: int) -> List[str]:
    print(response)
    """解析批量响应"""

    content = response["choices"][0]["message"]["content"]
    content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE)
    print(content)
    if content:
        # 按问题序号分割答案，需要根据实际返回格式调整
        answers = [""] * batch_size
        try:
            items = json.loads(content)  # 将JSON字符串转换为Python列表
            answers = [item['answer'] for item in items]
        except json.JSONDecodeError as e:
            print("JSON解析错误:", e)
        except KeyError as e:
            print("键不存在:", e)
        # answers = re.split(r'\d+\. Ans:\s*', content)[1:]
        return answers
    return [""] * batch_size


def get_deepseek_answer(question: str, max_retries=3) -> str:
    """调用DeepSeek API获取答案"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}{question}"}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=HEADERS, json=payload, timeout=30)
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"].strip()
            if answer:
                return answer
            return "Error: Empty response"

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2 ** attempt)  # 指数退避

    return "Error: API call failed"


def clean_answer(answer: str) -> str:
    """清洗回答内容"""
    # 去除可能的API错误信息
    if answer.startswith("Error:"):
        return ""

    # 标准化换行符
    answer = answer.replace("\r\n", "\n")

    # 去除多余空行
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    return answer.strip()


def build_dataset(questions: List[str]) -> List[Dict]:
    """构建问答数据集"""
    dataset = []

    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}: {question[:50]}...")

        answer = get_deepseek_answer(question)
        cleaned_answer = clean_answer(answer)

        if cleaned_answer:
            dataset.append({
                "instruction": INSTRUCTION,
                "input": question,
                "output": cleaned_answer
            })

        time.sleep(1)  # 控制请求速率
    return dataset


def build_batched_dataset(group_questions: dict) -> List[Dict]:
    """构建问答数据集"""
    dataset = []
    for parent, items in group_questions.items():
        batch_questions = [item[1] for item in items]
        batch_answers = batch_get_answers(batch_questions, min(len(batch_questions), 2))

        # batch_answers = batch_get_answers(batch_questions, len(batch_questions))
        print(f"Length of batch_answers:{len(batch_answers)}, Length of batch_questions:{len(batch_questions)}")
        print(f"batch_answers:{batch_answers}")

        if len(batch_answers) > 0:
            for ques, ans in zip(batch_questions, batch_answers):
                dataset.append({
                    "instruction": INSTRUCTION,
                    "input": ques,
                    "output": ans
                })
    return dataset


def save_dataset(dataset: List[Dict], output_path: str):
    """保存数据集到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {Path(output_path).resolve()}")


def main():
    # 步骤1：解析问题
    questions = extract_questions(MARKDOWN_PATH)
    print(f"Found {len(questions)} unique questions")
    print(f"======================================\n{questions}")
    grouped = group_questions(questions)
    print_group_queation_tree(grouped)
    # 步骤2：生成问答对
    # dataset = build_dataset(questions)
    dataset = build_batched_dataset(grouped)
    # 步骤3：数据清洗
    # 去重（基于问题）
    seen = set()
    cleaned_dataset = []
    for item in dataset:
        if item["input"] not in seen and item["output"]:
            seen.add(item["input"])
            cleaned_dataset.append(item)

    print(f"Final dataset size after cleaning: {len(cleaned_dataset)}")

    # 步骤4：保存结果
    save_dataset(cleaned_dataset, OUTPUT_JSON)


if __name__ == "__main__":
    main()