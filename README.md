# DL_interview_qa_chatbot

本项目基于ChatGLM-6B,通过github上流行的深度学习面试八股文，解析构建自己的深度学习面试qa指令集，用于微调ChatGLM-6B. 本深度学习面试qa指令集intrview-qa.json包含1k+深度学习面试qa指令，前200条指令为种子指令，题目来源于[machine-learning-interview](https://github.com/zhengjingwei/machine-learning-interview#2-2-1), 问题的回答调用deepseek的api接口，采用deepseek-chat模型（详见[deepseek官网](https://api-docs.deepseek.com/zh-cn/)）获取问题回答, 后800多条指令通过构建类似[Self-Instruct](https://github.com/yizhongw/self-instruct)的pipeline进行构造，不同的是我们的api接口使用deepseek.
## 数据构建
基于[machine-learning-interview](https://github.com/zhengjingwei/machine-learning-interview#2-2-1)仓库整理的深度学习高频面试题，用data.py进行问题解析、deepseek API调用获取问题答案、构建种子数据集interview_qa.json
运行脚本
``
python data.py
``
获得200+条深度学习面试qa指令集

## 基于self-instruct方法扩展数据集[TODO]


## 对ChatGLM-6B进行微调训练
基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


## 用chatglm.cpp对微调后的ChatGLM-6B进行模型量化及部署
