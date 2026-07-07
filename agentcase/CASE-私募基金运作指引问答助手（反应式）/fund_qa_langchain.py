#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
私募基金运作指引问答助手 - 反应式智能体实现

适合反应式架构的私募基金问答助手，使用LangGraph实现主动思考和工具选择。
"""

import re
import os
from typing import List, Dict, Any, Annotated, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 通义千问 API 配置（密钥从系统环境变量读取）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = "qwen3.7-max"

# 简化的私募基金规则数据库
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]


@tool
def search_rules_by_keywords(keywords: str) -> str:
    """通过关键词搜索相关私募基金规则。输入应为相关关键词，多个关键词用逗号或空格分隔。"""
    keywords = keywords.strip().lower()
    keyword_list = re.split(r'[,，\s]+', keywords)

    matched_rules = []
    for rule in FUND_RULES_DB:
        rule_text = (rule["category"] + " " + rule["question"]).lower()
        match_count = sum(1 for kw in keyword_list if kw in rule_text)
        if match_count > 0:
            matched_rules.append((rule, match_count))

    matched_rules.sort(key=lambda x: x[1], reverse=True)

    if not matched_rules:
        return "未找到与关键词相关的规则。"

    result = []
    for rule, _ in matched_rules[:2]:
        result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")

    return "\n\n".join(result)


@tool
def search_rules_by_category(category: str) -> str:
    """根据规则类别查询私募基金规则。输入应为类别名称，可选类别：设立与募集、监管规定。"""
    category = category.strip()
    matched_rules = []

    for rule in FUND_RULES_DB:
        if category.lower() in rule["category"].lower():
            matched_rules.append(rule)

    if not matched_rules:
        return f"未找到类别为 '{category}' 的规则。"

    result = []
    for rule in matched_rules:
        result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")

    return "\n\n".join(result)


@tool
def answer_question(query: str) -> str:
    """在知识库中搜索并回答用户关于私募基金的问题。输入应为完整的用户问题。"""
    query = query.strip()

    best_rule = None
    best_score = 0

    for rule in FUND_RULES_DB:
        query_words = set(query.lower().split())
        rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
        common_words = query_words.intersection(rule_words)

        score = len(common_words) / max(1, len(query_words))
        if score > best_score:
            best_score = score
            best_rule = rule

    if best_score < 0.2 or best_rule is None:
        return "在知识库中未找到与该问题直接相关的信息。请尝试使用关键词搜索或类别查询。"

    return f"根据知识库信息：\n\n类别: {best_rule['category']}\n问题: {best_rule['question']}\n答案: {best_rule['answer']}"


def create_fund_qa_agent():
    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "未找到 API 密钥，请在 ~/.zshrc 中配置 DASHSCOPE_API_KEY 后执行 source ~/.zshrc"
        )

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )

    tools = [search_rules_by_keywords, search_rules_by_category, answer_question]

    system_prompt = """你是一个私募基金问答助手，专门回答关于私募基金规则和运作的问题。

你可以使用以下工具来查询信息：
1. search_rules_by_keywords: 通过关键词搜索相关规则
2. search_rules_by_category: 按类别查询规则（类别：设立与募集、监管规定）
3. answer_question: 直接搜索并回答问题

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确"""

    agent = create_agent(llm, tools, system_prompt=system_prompt)

    return agent


if __name__ == "__main__":
    fund_qa_agent = create_fund_qa_agent()

    print("=== 私募基金运作指引问答助手（反应式智能体）===\n")
    print(f"使用模型：{QWEN_MODEL}")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")

    while True:
        user_input = input("请输入您的问题：")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break

        response = fund_qa_agent.invoke({"messages": [HumanMessage(content=user_input)]})

        # 获取最后一条AI消息作为回答
        for msg in reversed(response["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"回答: {msg.content}\n")
                break

        print("-" * 40)
