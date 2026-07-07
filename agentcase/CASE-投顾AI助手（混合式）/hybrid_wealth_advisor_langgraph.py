#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手

基于LangGraph实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过协调层动态切换处理模式，提供智能化财富管理咨询服务。

三层架构：
1. 底层（反应式）：即时响应客户查询，提供快速反馈
2. 中层（协调）：评估任务类型和优先级，动态选择处理模式
3. 顶层（深思熟虑）：进行复杂的投资分析和长期财务规划
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Literal, TypedDict, Optional, Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import warnings
warnings.filterwarnings("ignore")

# 设置API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 创建LLM实例（使用ChatTongyi以支持工具调用）
llm = ChatTongyi(model_name="qwen-turbo-latest", dashscope_api_key=DASHSCOPE_API_KEY)

# 定义客户信息数据结构
class CustomerProfile(BaseModel):
    """客户画像信息"""
    customer_id: str = Field(..., description="客户ID")
    risk_tolerance: Literal["保守型", "稳健型", "平衡型", "成长型", "进取型"] = Field(..., description="风险承受能力")
    investment_horizon: Literal["短期", "中期", "长期"] = Field(..., description="投资期限")
    financial_goals: List[str] = Field(..., description="财务目标")
    investment_preferences: List[str] = Field(..., description="投资偏好")
    portfolio_value: float = Field(..., description="投资组合总价值")
    current_allocations: Dict[str, float] = Field(..., description="当前资产配置")

# 定义工具
@tool
def query_shanghai_index() -> str:
    """查询上证指数实时行情，获取当前点位、涨跌和涨跌幅信息"""
    name = "上证指数"
    price = "3125.62"
    change = "6.32"
    pct = "0.20"
    result = f"{name} 当前点位: {price}，涨跌: {change}，涨跌幅: {pct}%"
    print(f"[工具调用] {result}")
    return result

@tool
def query_portfolio_allocation(asset_type: str) -> str:
    """查询客户投资组合中特定资产类型的配置比例

    Args:
        asset_type: 资产类型，如"股票"、"债券"、"现金"、"另类投资"
    """
    # 模拟数据
    allocations = {
        "股票": "40%",
        "债券": "30%",
        "现金": "10%",
        "另类投资": "20%"
    }
    result = allocations.get(asset_type, f"未找到{asset_type}的配置信息")
    print(f"[工具调用] 查询{asset_type}配置: {result}")
    return f"{asset_type}在投资组合中的配置比例为: {result}"

@tool
def query_market_news() -> str:
    """查询最新市场新闻和动态"""
    news = [
        "央行维持利率不变，市场预期稳定",
        "科技板块持续走强，半导体行业领涨",
        "外资连续三日净流入A股市场"
    ]
    result = "最新市场动态:\n" + "\n".join(f"- {n}" for n in news)
    print(f"[工具调用] {result}")
    return result

# 工具列表
tools = [query_shanghai_index, query_portfolio_allocation, query_market_news]

# 绑定工具到LLM
llm_with_tools = llm.bind_tools(tools)

# 创建工具节点
tool_node = ToolNode(tools)

# 定义状态类型
class WealthAdvisorState(TypedDict):
    """财富顾问智能体的状态"""
    # 输入
    user_query: str
    customer_profile: Optional[Dict[str, Any]]

    # 处理状态
    query_type: Optional[Literal["emergency", "informational", "analytical"]]
    processing_mode: Optional[Literal["reactive", "deliberative"]]
    market_data: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]

    # 消息历史（用于工具调用）
    messages: Annotated[List[BaseMessage], add_messages]

    # 输出
    final_response: Optional[str]

    # 控制流
    current_phase: Optional[str]
    error: Optional[str]

# 提示模板
ASSESSMENT_PROMPT = """你是一个财富管理投顾AI助手的协调层。请评估以下用户查询，确定其类型和应该采用的处理模式。

用户查询: {user_query}

请判断:
1. 查询类型:
   - "emergency": 紧急的或直接的查询，需要立即响应（如市场状况、账户信息、产品信息等）
   - "informational": 信息性的查询，需要特定领域知识（如税务政策、投资工具介绍等）
   - "analytical": 需要深度分析的查询（如投资组合优化、长期理财规划等）

2. 建议的处理模式:
   - "reactive": 适用于需要快速反应的查询
   - "deliberative": 适用于需要深度思考和分析的查询

请以JSON格式返回结果，包含以下字段:
- query_type: 查询类型（上述三种类型之一）
- processing_mode: 处理模式（上述两种模式之一）
- reasoning: 决策理由的简要说明
"""

DATA_COLLECTION_PROMPT = """你是一个财富管理投顾AI助手的数据收集模块。基于以下用户查询，确定需要收集哪些市场和财务数据进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

请确定需要收集的数据类型，例如:
- 资产类别表现数据
- 经济指标
- 行业趋势
- 历史回报率
- 风险指标
- 税收信息
- 其他相关数据

以JSON格式返回结果，包含以下字段:
- required_data_types: 需要收集的数据类型列表
- data_sources: 建议的数据来源列表
- collected_data: 模拟收集的数据（为简化示例，请生成合理的模拟数据）
"""

ANALYSIS_PROMPT = """你是一个财富管理投顾AI助手的分析引擎。请根据收集的数据对用户的投资情况进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

市场数据:
{market_data}

请提供全面的投资分析，包括:
1. 当前市场状况评估
2. 客户投资组合分析
3. 个性化投资建议
4. 风险评估
5. 预期结果和回报预测

以JSON格式返回分析结果，包含以下字段:
- market_assessment: 市场评估
- portfolio_analysis: 投资组合分析
- recommendations: 投资建议列表
- risk_analysis: 风险分析
- expected_outcomes: 预期结果
"""

RECOMMENDATION_PROMPT = """你是一个财富管理投顾AI助手。请根据深入分析结果，为客户准备最终的咨询建议。

用户查询: {user_query}

客户信息:
{customer_profile}

分析结果:
{analysis_results}

请提供专业、个性化且详细的投资建议，语言应友好易懂，避免过多专业术语。建议应包括:
1. 总体投资策略
2. 具体行动步骤
3. 资产配置建议
4. 风险管理策略
5. 时间框架
6. 预期收益
7. 后续跟进计划

返回格式应为自然语言文本，适合直接呈现给客户。
"""

# 第一阶段：情境评估 - 确定查询类型和处理模式
def assess_query(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] 进入节点: assess_query")

    prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"user_query": state["user_query"]})

    print(f"[DEBUG] LLM评估输出: {result}")

    processing_mode = result.get("processing_mode", "reactive")
    if processing_mode not in ["reactive", "deliberative"]:
        processing_mode = "reactive"

    query_type = result.get("query_type", "emergency")
    if query_type not in ["emergency", "informational", "analytical"]:
        query_type = "emergency"

    print(f"[DEBUG] 分支判断: processing_mode={processing_mode}, query_type={query_type}")

    return {
        "query_type": query_type,
        "processing_mode": processing_mode,
    }

# 反应式处理 - 初始化消息并调用带工具的LLM
def reactive_agent(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] 进入节点: reactive_agent")

    customer_info = json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2)

    system_prompt = f"""你是一个专业的财富管理投顾AI助手，请根据用户的问题提供专业、简洁、准确的回答。

客户信息:
{customer_info}

你可以使用以下工具来获取实时数据:
- query_shanghai_index: 查询上证指数实时行情
- query_portfolio_allocation: 查询投资组合中特定资产的配置比例
- query_market_news: 查询最新市场新闻

请根据用户问题判断是否需要调用工具，如果需要请调用相应工具获取数据后再回答。"""

    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=f"{system_prompt}\n\n用户问题: {state['user_query']}")]

    response = llm_with_tools.invoke(messages)
    print(f"[DEBUG] LLM响应: {response}")

    return {"messages": [response]}

# 判断是否需要继续调用工具
def should_continue_tools(state: WealthAdvisorState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    # 检查是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[DEBUG] 检测到工具调用: {last_message.tool_calls}")
        return "tools"

    return "end"

# 从消息中提取最终响应
def extract_reactive_response(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] 进入节点: extract_reactive_response")

    messages = state.get("messages", [])

    # 找到最后一条AI消息作为最终响应
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return {"final_response": msg.content}

    return {"final_response": "无法生成响应"}

# 数据收集 - 收集进行深度分析所需的数据
def collect_data(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] 进入节点: collect_data")

    prompt = ChatPromptTemplate.from_template(DATA_COLLECTION_PROMPT)
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "customer_profile": json.dumps(state.get("customer_profile", ), ensure_ascii=False, indent=2)
    })

    return {
        "market_data": result.get("collected_data", {}),
        "current_phase": "analyze"
    }

# 深度分析 - 分析数据和客户情况
def analyze_data(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] 进入节点: analyze_data")

    prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2),
        "market_data": json.dumps(state.get("market_data", {}), ensure_ascii=False, indent=2)
    })

    return {
        "analysis_results": result,
        "current_phase": "recommend"
    }

# 生成建议 - 根据分析结果提供投资建议
def generate_recommendations(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] 进入节点: generate_recommendations")

    prompt = ChatPromptTemplate.from_template(RECOMMENDATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2),
        "analysis_results": json.dumps(state.get("analysis_results", {}), ensure_ascii=False, indent=2)
    })

    return {
        "final_response": result,
        "current_phase": "respond"
    }

# 创建智能体工作流
def create_wealth_advisor_workflow() -> StateGraph:
    """创建财富顾问混合智能体工作流"""

    workflow = StateGraph(WealthAdvisorState)

    # 添加节点
    workflow.add_node("assess", assess_query)
    workflow.add_node("reactive_agent", reactive_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_response", extract_reactive_response)
    workflow.add_node("collect_data", collect_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("recommend", generate_recommendations)

    # 设置入口点
    workflow.set_entry_point("assess")

    # 评估后的分支路由
    workflow.add_conditional_edges(
        "assess",
        lambda x: "reactive_agent" if x.get("processing_mode") == "reactive" else "collect_data",
        {
            "reactive_agent": "reactive_agent",
            "collect_data": "collect_data"
        }
    )

    # 反应式Agent的工具调用循环
    workflow.add_conditional_edges(
        "reactive_agent",
        should_continue_tools,
        {
            "tools": "tools",
            "end": "extract_response"
        }
    )

    # 工具执行后返回Agent继续处理
    workflow.add_edge("tools", "reactive_agent")

    # 提取响应后结束
    workflow.add_edge("extract_response", END)

    # 深思熟虑模式的流程
    workflow.add_edge("collect_data", "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", END)

    return workflow.compile()

# 示例客户画像数据
SAMPLE_CUSTOMER_PROFILES = {
    "customer1": {
        "customer_id": "C10012345",
        "risk_tolerance": "平衡型",
        "investment_horizon": "中期",
        "financial_goals": ["退休规划", "子女教育金"],
        "investment_preferences": ["ESG投资", "科技行业"],
        "portfolio_value": 1500000.0,
        "current_allocations": {
            "股票": 0.40,
            "债券": 0.30,
            "现金": 0.10,
            "另类投资": 0.20
        }
    },
    "customer2": {
        "customer_id": "C10067890",
        "risk_tolerance": "进取型",
        "investment_horizon": "长期",
        "financial_goals": ["财富增长", "资产配置多元化"],
        "investment_preferences": ["新兴市场", "高成长行业"],
        "portfolio_value": 3000000.0,
        "current_allocations": {
            "股票": 0.65,
            "债券": 0.15,
            "现金": 0.05,
            "另类投资": 0.15
        }
    }
}

# 运行智能体
def run_wealth_advisor(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """运行财富顾问智能体并返回结果"""

    agent = create_wealth_advisor_workflow()
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])

    initial_state = {
        "user_query": user_query,
        "customer_profile": customer_profile,
        "query_type": None,
        "processing_mode": None,
        "market_data": None,
        "analysis_results": None,
        "messages": [],
        "final_response": None,
        "current_phase": "assess",
        "error": None
    }

    print("LangGraph Mermaid流程图：")
    print(agent.get_graph().draw_mermaid())

    result = agent.invoke(initial_state)
    return result

# 主函数
if __name__ == "__main__":
    print("=== 混合智能体 - 财富管理投顾AI助手 ===\n")
    print("使用模型：Qwen-Turbo-Latest\n")
    print("\n" + "-"*50 + "\n")

    SAMPLE_QUERIES = [
        # 紧急/简单查询 - 适合反应式处理
        "今天上证指数的表现如何？",
        "我的投资组合中股票占比是多少？",
        "请解释一下什么是ETF？",

        # 分析性查询 - 适合深思熟虑处理
        "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
        "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
        "我想为子女准备教育金，请帮我设计一个10年期的投资计划。"
    ]

    print("请选择一个示例查询或输入您自己的查询:\n")
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"{i}. {query}")
    print("0. 输入自定义查询")

    choice = input("\n请输入选项数字(0-6): ")

    if choice == "0":
        user_query = input("请输入您的查询: ")
    else:
        idx = int(choice) - 1
        if 0 <= idx < len(SAMPLE_QUERIES):
            user_query = SAMPLE_QUERIES[idx]
        else:
            print("无效选择，使用默认查询")
            user_query = SAMPLE_QUERIES[0]

    customer_id = "customer1"
    customer_choice = input("\n选择客户 (1: 平衡型投资者, 2: 进取型投资者): ")
    if customer_choice == "2":
        customer_id = "customer2"

    print(f"\n用户查询: {user_query}")
    print(f"选择客户: {SAMPLE_CUSTOMER_PROFILES[customer_id]['risk_tolerance']} 投资者")
    print("\n正在处理...\n")

    start_time = datetime.now()
    result = run_wealth_advisor(user_query, customer_id)
    end_time = datetime.now()

    process_mode = result.get("processing_mode", "未知")
    if process_mode == "reactive":
        print("【处理模式: 反应式】- 快速响应简单查询")
    else:
        print("【处理模式: 深思熟虑】- 深度分析复杂查询")

    print("\n=== 响应结果 ===\n")
    print(result.get("final_response", "未生成响应"))

    process_time = (end_time - start_time).total_seconds()
    print(f"\n处理用时: {process_time:.2f}秒")
