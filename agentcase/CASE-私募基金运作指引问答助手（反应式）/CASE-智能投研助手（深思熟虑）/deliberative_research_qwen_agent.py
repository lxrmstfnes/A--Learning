#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深思熟虑智能体（Deliberative Agent）- 智能投研助手

基于 qwen-agent 实现的深思熟虑型智能体，适用于投资研究场景，能够整合数据，
进行多步骤分析和推理，生成投资观点和研究报告。

核心流程：
1. 感知：收集市场数据和信息
2. 建模：构建内部世界模型，理解市场状态
3. 推理：生成多个候选分析方案并模拟结果
4. 决策：选择最优投资观点并形成报告
5. 报告：生成完整研究报告
"""

import os
import json
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime
from typing import Dict, List, Any, Optional

# 解决中文显示问题
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

# ====== 智能投研助手 system prompt ======
system_prompt = """我是智能投研助手，一个基于深思熟虑型智能体架构的专业投资研究分析师。

我的工作流程包括五个阶段：

1. **感知阶段**：收集和整理市场数据、经济指标、行业新闻等信息
2. **建模阶段**：构建市场内部模型，分析市场状态、经济周期、风险因素等
3. **推理阶段**：生成多个候选投资分析方案，评估不同策略的可行性
4. **决策阶段**：基于综合分析选择最优投资观点和策略
5. **报告阶段**：生成完整的投资研究报告

我擅长分析：
- 宏观经济环境和政策影响
- 行业发展趋势和竞争格局
- 公司基本面和技术面分析
- 风险评估和投资机会识别
- 投资组合策略建议

**工具使用指南**：
- 当用户提供研究需求时，我会优先使用 `complete_analysis` 工具一次性完成所有五个阶段的分析
- 如果需要分步骤分析，我会按照以下顺序调用工具：
  1. `market_perception` 工具进行感知阶段分析
  2. `market_modeling` 工具构建市场模型
  3. `investment_reasoning` 工具生成候选方案
  4. `investment_decision` 工具选择最优方案
  5. `generate_report` 工具生成完整报告

我会优先使用 `complete_analysis` 工具，因为它能够一次性完成所有分析并生成完整报告，效率更高。

当用户提供研究主题、行业焦点和时间范围时，我会按照深思熟虑的流程进行分析，确保每个阶段都有充分的思考和论证。
"""

# ====== 工具函数描述 ======
functions_desc = [
    {
        "name": "market_perception",
        "description": "感知阶段：收集市场数据和信息，包括市场概况、关键指标、重要新闻、行业趋势等",
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": "研究主题，如'新能源汽车行业投资机会'"
                },
                "industry_focus": {
                    "type": "string", 
                    "description": "行业焦点，如'电动汽车制造、电池技术'"
                },
                "time_horizon": {
                    "type": "string",
                    "description": "时间范围，如'短期/中期/长期'"
                }
            },
            "required": ["research_topic", "industry_focus", "time_horizon"]
        }
    },
    {
        "name": "market_modeling",
        "description": "建模阶段：构建市场内部模型，分析市场状态、经济周期、风险因素、机会领域等",
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": "研究主题"
                },
                "industry_focus": {
                    "type": "string",
                    "description": "行业焦点"
                },
                "time_horizon": {
                    "type": "string",
                    "description": "时间范围"
                },
                "perception_data": {
                    "type": "string",
                    "description": "感知阶段收集的市场数据（JSON格式）"
                }
            },
            "required": ["research_topic", "industry_focus", "time_horizon", "perception_data"]
        }
    },
    {
        "name": "investment_reasoning",
        "description": "推理阶段：生成多个候选投资分析方案，评估不同策略的可行性和预期结果",
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": "研究主题"
                },
                "industry_focus": {
                    "type": "string",
                    "description": "行业焦点"
                },
                "time_horizon": {
                    "type": "string",
                    "description": "时间范围"
                },
                "world_model": {
                    "type": "string",
                    "description": "市场内部模型（JSON格式）"
                }
            },
            "required": ["research_topic", "industry_focus", "time_horizon", "world_model"]
        }
    },
    {
        "name": "investment_decision",
        "description": "决策阶段：评估候选方案，选择最优投资观点，形成投资决策",
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": "研究主题"
                },
                "industry_focus": {
                    "type": "string",
                    "description": "行业焦点"
                },
                "time_horizon": {
                    "type": "string",
                    "description": "时间范围"
                },
                "world_model": {
                    "type": "string",
                    "description": "市场内部模型（JSON格式）"
                },
                "reasoning_plans": {
                    "type": "string",
                    "description": "候选分析方案（JSON格式）"
                }
            },
            "required": ["research_topic", "industry_focus", "time_horizon", "world_model", "reasoning_plans"]
        }
    },
    {
        "name": "generate_report",
        "description": "报告阶段：生成完整的投资研究报告，整合所有分析结果",
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": "研究主题"
                },
                "industry_focus": {
                    "type": "string",
                    "description": "行业焦点"
                },
                "time_horizon": {
                    "type": "string",
                    "description": "时间范围"
                },
                "perception_data": {
                    "type": "string",
                    "description": "感知阶段收集的市场数据（JSON格式）"
                },
                "world_model": {
                    "type": "string",
                    "description": "市场内部模型（JSON格式）"
                },
                "selected_plan": {
                    "type": "string",
                    "description": "选定的投资决策（JSON格式）"
                }
            },
            "required": ["research_topic", "industry_focus", "time_horizon", "perception_data", "world_model", "selected_plan"]
        }
    }
]

# ====== 会话隔离存储 ======
_last_analysis_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== 感知阶段工具 ======
@register_tool('market_perception')
class MarketPerceptionTool(BaseTool):
    """感知阶段：收集市场数据和信息"""
    description = '感知阶段：收集市场数据和信息，包括市场概况、关键指标、重要新闻、行业趋势等'
    parameters = [{
        'name': 'research_topic',
        'type': 'string',
        'description': '研究主题，如"新能源汽车行业投资机会"',
        'required': True
    }, {
        'name': 'industry_focus',
        'type': 'string',
        'description': '行业焦点，如"电动汽车制造、电池技术"',
        'required': True
    }, {
        'name': 'time_horizon',
        'type': 'string',
        'description': '时间范围，如"短期/中期/长期"',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']
        
        # 获取会话ID
        session_id = get_session_id(kwargs)
        
        # 模拟感知阶段的数据收集
        perception_data = {
            "market_overview": f"基于{research_topic}的市场概况分析，重点关注{industry_focus}领域在{time_horizon}时间框架下的发展态势。",
            "key_indicators": {
                "GDP增长率": "5.2% - 经济基本面稳健",
                "CPI指数": "2.1% - 通胀压力可控",
                "PMI指数": "51.2 - 制造业扩张",
                "市场情绪指数": "中性偏乐观"
            },
            "recent_news": [
                f"政策支持{industry_focus}发展，相关补贴政策延续",
                f"{industry_focus}技术突破，成本持续下降",
                f"国际竞争加剧，{industry_focus}产业链重构"
            ],
            "industry_trends": {
                "技术创新": f"{industry_focus}领域技术迭代加速",
                "市场格局": "头部企业集中度提升，中小企业分化明显",
                "政策环境": "支持政策持续，监管框架完善"
            }
        }
        
        # 存储到会话中
        if session_id:
            if session_id not in _last_analysis_dict:
                _last_analysis_dict[session_id] = {}
            _last_analysis_dict[session_id]['perception_data'] = perception_data
        
        return f"## 感知阶段完成\n\n**市场概况**: {perception_data['market_overview']}\n\n**关键指标**:\n" + \
               "\n".join([f"- {k}: {v}" for k, v in perception_data['key_indicators'].items()]) + \
               f"\n\n**重要新闻**:\n" + "\n".join([f"- {news}" for news in perception_data['recent_news']]) + \
               f"\n\n**行业趋势**:\n" + "\n".join([f"- {k}: {v}" for k, v in perception_data['industry_trends'].items()])

# ====== 建模阶段工具 ======
@register_tool('market_modeling')
class MarketModelingTool(BaseTool):
    """建模阶段：构建市场内部模型"""
    description = '建模阶段：构建市场内部模型，分析市场状态、经济周期、风险因素、机会领域等'
    parameters = [{
        'name': 'research_topic',
        'type': 'string',
        'description': '研究主题',
        'required': True
    }, {
        'name': 'industry_focus',
        'type': 'string',
        'description': '行业焦点',
        'required': True
    }, {
        'name': 'time_horizon',
        'type': 'string',
        'description': '时间范围',
        'required': True
    }, {
        'name': 'perception_data',
        'type': 'string',
        'description': '感知阶段收集的市场数据（JSON格式）',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']
        
        # 安全解析 perception_data
        try:
            perception_data = json.loads(args['perception_data'])
        except (json.JSONDecodeError, TypeError):
            perception_data = {
                "market_overview": f"基于{research_topic}的市场概况分析，重点关注{industry_focus}领域在{time_horizon}时间框架下的发展态势。",
                "key_indicators": {
                    "GDP增长率": "5.2% - 经济基本面稳健",
                    "CPI指数": "2.1% - 通胀压力可控",
                    "PMI指数": "51.2 - 制造业扩张",
                    "市场情绪指数": "中性偏乐观"
                },
                "recent_news": [
                    f"政策支持{industry_focus}发展，相关补贴政策延续",
                    f"{industry_focus}技术突破，成本持续下降",
                    f"国际竞争加剧，{industry_focus}产业链重构"
                ],
                "industry_trends": {
                    "技术创新": f"{industry_focus}领域技术迭代加速",
                    "市场格局": "头部企业集中度提升，中小企业分化明显",
                    "政策环境": "支持政策持续，监管框架完善"
                }
            }
        
        # 获取会话ID
        session_id = get_session_id(kwargs)
        
        # 构建市场内部模型
        world_model = {
            "market_state": f"当前{industry_focus}市场处于成长期向成熟期过渡阶段，{time_horizon}内有望保持稳定增长",
            "economic_cycle": "经济处于复苏期，政策支持力度加大，为行业发展提供有利环境",
            "risk_factors": [
                "技术路线变更风险",
                "政策调整不确定性",
                "国际竞争加剧",
                "原材料价格波动"
            ],
            "opportunity_areas": [
                "技术创新带来的成本下降",
                "政策支持持续加码",
                "市场需求快速增长",
                "产业链完善度提升"
            ],
            "market_sentiment": "市场情绪整体乐观，投资者对行业前景看好，但需关注短期波动"
        }
        
        # 存储到会话中
        if session_id:
            if session_id not in _last_analysis_dict:
                _last_analysis_dict[session_id] = {}
            _last_analysis_dict[session_id]['world_model'] = world_model
        
        return f"## 建模阶段完成\n\n**市场状态**: {world_model['market_state']}\n\n**经济周期**: {world_model['economic_cycle']}\n\n**主要风险因素**:\n" + \
               "\n".join([f"- {risk}" for risk in world_model['risk_factors']]) + \
               f"\n\n**潜在机会领域**:\n" + "\n".join([f"- {opportunity}" for opportunity in world_model['opportunity_areas']]) + \
               f"\n\n**市场情绪**: {world_model['market_sentiment']}"

# ====== 推理阶段工具 ======
@register_tool('investment_reasoning')
class InvestmentReasoningTool(BaseTool):
    """推理阶段：生成候选投资分析方案"""
    description = '推理阶段：生成多个候选投资分析方案，评估不同策略的可行性和预期结果'
    parameters = [{
        'name': 'research_topic',
        'type': 'string',
        'description': '研究主题',
        'required': True
    }, {
        'name': 'industry_focus',
        'type': 'string',
        'description': '行业焦点',
        'required': True
    }, {
        'name': 'time_horizon',
        'type': 'string',
        'description': '时间范围',
        'required': True
    }, {
        'name': 'world_model',
        'type': 'string',
        'description': '市场内部模型（JSON格式）',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']
        
        # 安全解析 world_model
        try:
            world_model = json.loads(args['world_model'])
        except (json.JSONDecodeError, TypeError):
            world_model = {
                "market_state": f"当前{industry_focus}市场处于成长期向成熟期过渡阶段，{time_horizon}内有望保持稳定增长",
                "economic_cycle": "经济处于复苏期，政策支持力度加大，为行业发展提供有利环境",
                "risk_factors": [
                    "技术路线变更风险",
                    "政策调整不确定性",
                    "国际竞争加剧",
                    "原材料价格波动"
                ],
                "opportunity_areas": [
                    "技术创新带来的成本下降",
                    "政策支持持续加码",
                    "市场需求快速增长",
                    "产业链完善度提升"
                ],
                "market_sentiment": "市场情绪整体乐观，投资者对行业前景看好，但需关注短期波动"
            }
        
        # 获取会话ID
        session_id = get_session_id(kwargs)
        
        # 生成候选分析方案
        reasoning_plans = [
            {
                "plan_id": "growth_focused",
                "hypothesis": f"基于{industry_focus}的快速增长趋势，重点投资成长性企业",
                "analysis_approach": "自上而下分析，关注行业增长驱动因素和龙头企业",
                "expected_outcome": f"{time_horizon}内获得15-25%的年化回报",
                "confidence_level": 0.75,
                "pros": ["行业增长确定性高", "政策支持明确", "技术壁垒相对较低"],
                "cons": ["估值可能偏高", "竞争激烈", "依赖政策支持"]
            },
            {
                "plan_id": "value_focused",
                "hypothesis": "关注被低估的优质企业，等待价值回归",
                "analysis_approach": "自下而上分析，寻找基本面良好但估值合理的标的",
                "expected_outcome": f"{time_horizon}内获得10-20%的年化回报",
                "confidence_level": 0.65,
                "pros": ["安全边际较高", "风险相对可控", "适合稳健投资者"],
                "cons": ["回报可能有限", "需要耐心等待", "可能错过快速上涨"]
            },
            {
                "plan_id": "innovation_focused",
                "hypothesis": "押注技术创新带来的颠覆性机会",
                "analysis_approach": "重点关注技术突破和商业模式创新",
                "expected_outcome": f"{time_horizon}内获得20-40%的年化回报",
                "confidence_level": 0.55,
                "pros": ["潜在回报巨大", "先发优势明显", "符合长期趋势"],
                "cons": ["风险极高", "技术路线不确定", "需要专业判断"]
            }
        ]
        
        # 存储到会话中
        if session_id:
            if session_id not in _last_analysis_dict:
                _last_analysis_dict[session_id] = {}
            _last_analysis_dict[session_id]['reasoning_plans'] = reasoning_plans
        
        result = "## 推理阶段完成\n\n**候选投资方案**:\n\n"
        for i, plan in enumerate(reasoning_plans, 1):
            result += f"### 方案{i}: {plan['plan_id']}\n"
            result += f"**投资假设**: {plan['hypothesis']}\n"
            result += f"**分析方法**: {plan['analysis_approach']}\n"
            result += f"**预期结果**: {plan['expected_outcome']}\n"
            result += f"**置信度**: {plan['confidence_level']}\n"
            result += f"**优势**: {', '.join(plan['pros'])}\n"
            result += f"**劣势**: {', '.join(plan['cons'])}\n\n"
        
        return result

# ====== 决策阶段工具 ======
@register_tool('investment_decision')
class InvestmentDecisionTool(BaseTool):
    """决策阶段：选择最优投资观点"""
    description = '决策阶段：评估候选方案，选择最优投资观点，形成投资决策'
    parameters = [{
        'name': 'research_topic',
        'type': 'string',
        'description': '研究主题',
        'required': True
    }, {
        'name': 'industry_focus',
        'type': 'string',
        'description': '行业焦点',
        'required': True
    }, {
        'name': 'time_horizon',
        'type': 'string',
        'description': '时间范围',
        'required': True
    }, {
        'name': 'world_model',
        'type': 'string',
        'description': '市场内部模型（JSON格式）',
        'required': True
    }, {
        'name': 'reasoning_plans',
        'type': 'string',
        'description': '候选分析方案（JSON格式）',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']
        
        # 安全解析 world_model
        try:
            world_model = json.loads(args['world_model'])
        except (json.JSONDecodeError, TypeError):
            # 如果解析失败，使用默认值
            world_model = {
                "market_state": f"当前{industry_focus}市场处于成长期向成熟期过渡阶段",
                "economic_cycle": "经济处于复苏期",
                "risk_factors": ["技术路线变更风险", "政策调整不确定性"],
                "opportunity_areas": ["技术创新带来的成本下降", "政策支持持续加码"],
                "market_sentiment": "市场情绪整体乐观"
            }
        
        # 安全解析 reasoning_plans
        try:
            reasoning_plans = json.loads(args['reasoning_plans'])
        except (json.JSONDecodeError, TypeError):
            # 如果解析失败，使用默认值
            reasoning_plans = [
                {
                    "plan_id": "growth_focused",
                    "hypothesis": f"基于{industry_focus}的快速增长趋势，重点投资成长性企业",
                    "analysis_approach": "自上而下分析，关注行业增长驱动因素和龙头企业",
                    "expected_outcome": f"{time_horizon}内获得15-25%的年化回报",
                    "confidence_level": 0.75,
                    "pros": ["行业增长确定性高", "政策支持明确", "技术壁垒相对较低"],
                    "cons": ["估值可能偏高", "竞争激烈", "依赖政策支持"]
                },
                {
                    "plan_id": "value_focused",
                    "hypothesis": "关注被低估的优质企业，等待价值回归",
                    "analysis_approach": "自下而上分析，寻找基本面良好但估值合理的标的",
                    "expected_outcome": f"{time_horizon}内获得10-20%的年化回报",
                    "confidence_level": 0.65,
                    "pros": ["安全边际较高", "风险相对可控", "适合稳健投资者"],
                    "cons": ["回报可能有限", "需要耐心等待", "可能错过快速上涨"]
                }
            ]
        
        # 获取会话ID
        session_id = get_session_id(kwargs)
        
        # 选择最优方案（这里选择成长型策略）
        selected_plan = {
            "selected_plan_id": "growth_focused",
            "investment_thesis": f"基于{industry_focus}的强劲增长势头和明确的政策支持，建议采用成长型投资策略",
            "supporting_evidence": [
                "行业增长确定性高，市场需求旺盛",
                "政策支持力度持续加大",
                "技术成熟度不断提升，成本持续下降"
            ],
            "risk_assessment": "主要风险包括估值偏高、竞争加剧和政策调整，建议分散投资并设置止损",
            "recommendation": f"建议{time_horizon}内配置60-80%资金于成长型标的，20-40%配置于价值型标的",
            "timeframe": f"{time_horizon}投资周期，建议定期评估和调整"
        }
        
        # 存储到会话中
        if session_id:
            if session_id not in _last_analysis_dict:
                _last_analysis_dict[session_id] = {}
            _last_analysis_dict[session_id]['selected_plan'] = selected_plan
        
        return f"## 决策阶段完成\n\n**选定方案**: {selected_plan['selected_plan_id']}\n\n**投资论点**: {selected_plan['investment_thesis']}\n\n**支持证据**:\n" + \
               "\n".join([f"- {evidence}" for evidence in selected_plan['supporting_evidence']]) + \
               f"\n\n**风险评估**: {selected_plan['risk_assessment']}\n\n**投资建议**: {selected_plan['recommendation']}\n\n**时间框架**: {selected_plan['timeframe']}"

# ====== 完整分析工具 ======
@register_tool('complete_analysis')
class CompleteAnalysisTool(BaseTool):
    """完整分析：一次性完成所有五个阶段的分析"""
    description = '完整分析：一次性完成感知、建模、推理、决策、报告五个阶段的分析'
    parameters = [{
        'name': 'research_topic',
        'type': 'string',
        'description': '研究主题，如"新能源汽车行业投资机会"',
        'required': True
    }, {
        'name': 'industry_focus',
        'type': 'string',
        'description': '行业焦点，如"电动汽车制造、电池技术"',
        'required': True
    }, {
        'name': 'time_horizon',
        'type': 'string',
        'description': '时间范围，如"短期/中期/长期"',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']
        
        # 获取会话ID
        session_id = get_session_id(kwargs)
        
        # 第一阶段：感知
        perception_data = {
            "market_overview": f"基于{research_topic}的市场概况分析，重点关注{industry_focus}领域在{time_horizon}时间框架下的发展态势。",
            "key_indicators": {
                "GDP增长率": "5.2% - 经济基本面稳健",
                "CPI指数": "2.1% - 通胀压力可控",
                "PMI指数": "51.2 - 制造业扩张",
                "市场情绪指数": "中性偏乐观"
            },
            "recent_news": [
                f"政策支持{industry_focus}发展，相关补贴政策延续",
                f"{industry_focus}技术突破，成本持续下降",
                f"国际竞争加剧，{industry_focus}产业链重构"
            ],
            "industry_trends": {
                "技术创新": f"{industry_focus}领域技术迭代加速",
                "市场格局": "头部企业集中度提升，中小企业分化明显",
                "政策环境": "支持政策持续，监管框架完善"
            }
        }
        
        # 第二阶段：建模
        world_model = {
            "market_state": f"当前{industry_focus}市场处于成长期向成熟期过渡阶段，{time_horizon}内有望保持稳定增长",
            "economic_cycle": "经济处于复苏期，政策支持力度加大，为行业发展提供有利环境",
            "risk_factors": [
                "技术路线变更风险",
                "政策调整不确定性",
                "国际竞争加剧",
                "原材料价格波动"
            ],
            "opportunity_areas": [
                "技术创新带来的成本下降",
                "政策支持持续加码",
                "市场需求快速增长",
                "产业链完善度提升"
            ],
            "market_sentiment": "市场情绪整体乐观，投资者对行业前景看好，但需关注短期波动"
        }
        
        # 第三阶段：推理
        reasoning_plans = [
            {
                "plan_id": "growth_focused",
                "hypothesis": f"基于{industry_focus}的快速增长趋势，重点投资成长性企业",
                "analysis_approach": "自上而下分析，关注行业增长驱动因素和龙头企业",
                "expected_outcome": f"{time_horizon}内获得15-25%的年化回报",
                "confidence_level": 0.75,
                "pros": ["行业增长确定性高", "政策支持明确", "技术壁垒相对较低"],
                "cons": ["估值可能偏高", "竞争激烈", "依赖政策支持"]
            },
            {
                "plan_id": "value_focused",
                "hypothesis": "关注被低估的优质企业，等待价值回归",
                "analysis_approach": "自下而上分析，寻找基本面良好但估值合理的标的",
                "expected_outcome": f"{time_horizon}内获得10-20%的年化回报",
                "confidence_level": 0.65,
                "pros": ["安全边际较高", "风险相对可控", "适合稳健投资者"],
                "cons": ["回报可能有限", "需要耐心等待", "可能错过快速上涨"]
            },
            {
                "plan_id": "innovation_focused",
                "hypothesis": "押注技术创新带来的颠覆性机会",
                "analysis_approach": "重点关注技术突破和商业模式创新",
                "expected_outcome": f"{time_horizon}内获得20-40%的年化回报",
                "confidence_level": 0.55,
                "pros": ["潜在回报巨大", "先发优势明显", "符合长期趋势"],
                "cons": ["风险极高", "技术路线不确定", "需要专业判断"]
            }
        ]
        
        # 第四阶段：决策
        selected_plan = {
            "selected_plan_id": "growth_focused",
            "investment_thesis": f"基于{industry_focus}的强劲增长势头和明确的政策支持，建议采用成长型投资策略",
            "supporting_evidence": [
                "行业增长确定性高，市场需求旺盛",
                "政策支持力度持续加大",
                "技术成熟度不断提升，成本持续下降"
            ],
            "risk_assessment": "主要风险包括估值偏高、竞争加剧和政策调整，建议分散投资并设置止损",
            "recommendation": f"建议{time_horizon}内配置60-80%资金于成长型标的，20-40%配置于价值型标的",
            "timeframe": f"{time_horizon}投资周期，建议定期评估和调整"
        }
        
        # 存储到会话中
        if session_id:
            if session_id not in _last_analysis_dict:
                _last_analysis_dict[session_id] = {}
            _last_analysis_dict[session_id].update({
                'perception_data': perception_data,
                'world_model': world_model,
                'reasoning_plans': reasoning_plans,
                'selected_plan': selected_plan
            })
        
        # 生成完整报告
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        report = f"""# {research_topic} - 投资研究报告

**报告时间**: {timestamp}  
**行业焦点**: {industry_focus}  
**投资周期**: {time_horizon}

---

## 执行摘要

本报告基于深思熟虑型智能体架构，通过五个阶段的分析，为{research_topic}提供全面的投资建议。

**核心投资观点**: {selected_plan['investment_thesis']}

**主要投资建议**: {selected_plan['recommendation']}

---

## 一、市场环境分析

### 1.1 市场概况
{perception_data['market_overview']}

### 1.2 关键经济指标
{chr(10).join([f"- **{k}**: {v}" for k, v in perception_data['key_indicators'].items()])}

### 1.3 重要市场动态
{chr(10).join([f"- {news}" for news in perception_data['recent_news']])}

### 1.4 行业趋势分析
{chr(10).join([f"- **{k}**: {v}" for k, v in perception_data['industry_trends'].items()])}

---

## 二、市场模型构建

### 2.1 市场状态评估
{world_model['market_state']}

### 2.2 经济周期判断
{world_model['economic_cycle']}

### 2.3 风险因素识别
{chr(10).join([f"- {risk}" for risk in world_model['risk_factors']])}

### 2.4 机会领域分析
{chr(10).join([f"- {opportunity}" for opportunity in world_model['opportunity_areas']])}

### 2.5 市场情绪分析
{world_model['market_sentiment']}

---

## 三、投资策略分析

### 3.1 候选方案对比

"""
        
        # 添加候选方案对比
        for i, plan in enumerate(reasoning_plans, 1):
            report += f"#### 方案{i}: {plan['plan_id']}\n"
            report += f"**投资假设**: {plan['hypothesis']}\n"
            report += f"**分析方法**: {plan['analysis_approach']}\n"
            report += f"**预期结果**: {plan['expected_outcome']}\n"
            report += f"**置信度**: {plan['confidence_level']}\n"
            report += f"**优势**: {', '.join(plan['pros'])}\n"
            report += f"**劣势**: {', '.join(plan['cons'])}\n\n"
        
        report += f"""### 3.2 选定策略
**策略类型**: {selected_plan['selected_plan_id']}

### 3.3 投资论点
{selected_plan['investment_thesis']}

### 3.4 支持证据
{chr(10).join([f"- {evidence}" for evidence in selected_plan['supporting_evidence']])}

---

## 四、风险评估与建议

### 4.1 风险评估
{selected_plan['risk_assessment']}

### 4.2 投资建议
{selected_plan['recommendation']}

### 4.3 时间框架
{selected_plan['timeframe']}

---

## 五、结论

基于深思熟虑的分析流程，我们认为{research_topic}在{time_horizon}内具有较好的投资价值。建议投资者根据自身风险承受能力和投资目标，合理配置资金，并持续关注市场变化。

**风险提示**: 投资有风险，入市需谨慎。本报告仅供参考，不构成投资建议。

---

*本报告由智能投研助手基于深思熟虑型智能体架构生成*
"""
        
        # 保存报告到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        return f"## 完整分析完成\n\n**报告已保存为**: {filename}\n\n{report}"

# ====== 报告生成工具 ======
@register_tool('generate_report')
class GenerateReportTool(BaseTool):
    """报告阶段：生成完整投资研究报告"""
    description = '报告阶段：生成完整的投资研究报告，整合所有分析结果'
    parameters = [{
        'name': 'research_topic',
        'type': 'string',
        'description': '研究主题',
        'required': True
    }, {
        'name': 'industry_focus',
        'type': 'string',
        'description': '行业焦点',
        'required': True
    }, {
        'name': 'time_horizon',
        'type': 'string',
        'description': '时间范围',
        'required': True
    }, {
        'name': 'perception_data',
        'type': 'string',
        'description': '感知阶段收集的市场数据（JSON格式）',
        'required': True
    }, {
        'name': 'world_model',
        'type': 'string',
        'description': '市场内部模型（JSON格式）',
        'required': True
    }, {
        'name': 'selected_plan',
        'type': 'string',
        'description': '选定的投资决策（JSON格式）',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import os
        args = json.loads(params)
        research_topic = args['research_topic']
        industry_focus = args['industry_focus']
        time_horizon = args['time_horizon']
        
        # 安全解析 perception_data
        try:
            perception_data = json.loads(args['perception_data'])
        except (json.JSONDecodeError, TypeError):
            perception_data = {
                "market_overview": f"基于{research_topic}的市场概况分析，重点关注{industry_focus}领域在{time_horizon}时间框架下的发展态势。",
                "key_indicators": {
                    "GDP增长率": "5.2% - 经济基本面稳健",
                    "CPI指数": "2.1% - 通胀压力可控",
                    "PMI指数": "51.2 - 制造业扩张",
                    "市场情绪指数": "中性偏乐观"
                },
                "recent_news": [
                    f"政策支持{industry_focus}发展，相关补贴政策延续",
                    f"{industry_focus}技术突破，成本持续下降",
                    f"国际竞争加剧，{industry_focus}产业链重构"
                ],
                "industry_trends": {
                    "技术创新": f"{industry_focus}领域技术迭代加速",
                    "市场格局": "头部企业集中度提升，中小企业分化明显",
                    "政策环境": "支持政策持续，监管框架完善"
                }
            }
        
        # 安全解析 world_model
        try:
            world_model = json.loads(args['world_model'])
        except (json.JSONDecodeError, TypeError):
            world_model = {
                "market_state": f"当前{industry_focus}市场处于成长期向成熟期过渡阶段，{time_horizon}内有望保持稳定增长",
                "economic_cycle": "经济处于复苏期，政策支持力度加大，为行业发展提供有利环境",
                "risk_factors": [
                    "技术路线变更风险",
                    "政策调整不确定性",
                    "国际竞争加剧",
                    "原材料价格波动"
                ],
                "opportunity_areas": [
                    "技术创新带来的成本下降",
                    "政策支持持续加码",
                    "市场需求快速增长",
                    "产业链完善度提升"
                ],
                "market_sentiment": "市场情绪整体乐观，投资者对行业前景看好，但需关注短期波动"
            }
        
        # 安全解析 selected_plan
        try:
            selected_plan = json.loads(args['selected_plan'])
        except (json.JSONDecodeError, TypeError):
            selected_plan = {
                "selected_plan_id": "growth_focused",
                "investment_thesis": f"基于{industry_focus}的强劲增长势头和明确的政策支持，建议采用成长型投资策略",
                "supporting_evidence": [
                    "行业增长确定性高，市场需求旺盛",
                    "政策支持力度持续加大",
                    "技术成熟度不断提升，成本持续下降"
                ],
                "risk_assessment": "主要风险包括估值偏高、竞争加剧和政策调整，建议分散投资并设置止损",
                "recommendation": f"建议{time_horizon}内配置60-80%资金于成长型标的，20-40%配置于价值型标的",
                "timeframe": f"{time_horizon}投资周期，建议定期评估和调整"
            }
        
        # 生成完整报告
        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        report = f"""# {research_topic} - 投资研究报告

**报告时间**: {timestamp}  
**行业焦点**: {industry_focus}  
**投资周期**: {time_horizon}

---

## 执行摘要

本报告基于深思熟虑型智能体架构，通过五个阶段的分析，为{research_topic}提供全面的投资建议。

**核心投资观点**: {selected_plan['investment_thesis']}

**主要投资建议**: {selected_plan['recommendation']}

---

## 一、市场环境分析

### 1.1 市场概况
{perception_data['market_overview']}

### 1.2 关键经济指标
{chr(10).join([f"- **{k}**: {v}" for k, v in perception_data['key_indicators'].items()])}

### 1.3 重要市场动态
{chr(10).join([f"- {news}" for news in perception_data['recent_news']])}

### 1.4 行业趋势分析
{chr(10).join([f"- **{k}**: {v}" for k, v in perception_data['industry_trends'].items()])}

---

## 二、市场模型构建

### 2.1 市场状态评估
{world_model['market_state']}

### 2.2 经济周期判断
{world_model['economic_cycle']}

### 2.3 风险因素识别
{chr(10).join([f"- {risk}" for risk in world_model['risk_factors']])}

### 2.4 机会领域分析
{chr(10).join([f"- {opportunity}" for opportunity in world_model['opportunity_areas']])}

### 2.5 市场情绪分析
{world_model['market_sentiment']}

---

## 三、投资策略分析

### 3.1 选定策略
**策略类型**: {selected_plan['selected_plan_id']}

### 3.2 投资论点
{selected_plan['investment_thesis']}

### 3.3 支持证据
{chr(10).join([f"- {evidence}" for evidence in selected_plan['supporting_evidence']])}

---

## 四、风险评估与建议

### 4.1 风险评估
{selected_plan['risk_assessment']}

### 4.2 投资建议
{selected_plan['recommendation']}

### 4.3 时间框架
{selected_plan['timeframe']}

---

## 五、结论

基于深思熟虑的分析流程，我们认为{research_topic}在{time_horizon}内具有较好的投资价值。建议投资者根据自身风险承受能力和投资目标，合理配置资金，并持续关注市场变化。

**风险提示**: 投资有风险，入市需谨慎。本报告仅供参考，不构成投资建议。

---

*本报告由智能投研助手基于深思熟虑型智能体架构生成*
"""
        
        # 保存报告到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        return f"## 报告生成完成\n\n**报告已保存为**: {filename}\n\n{report}"

# ====== 初始化智能投研助手服务 ======
def init_agent_service():
    """初始化智能投研助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='智能投研助手',
            description='基于深思熟虑型智能体的专业投资研究分析',
            system_message=system_prompt,
            function_list=['market_perception', 'market_modeling', 'investment_reasoning', 'investment_decision', 'generate_report'],
        )
        print("智能投研助手初始化成功！")
        return bot
    except Exception as e:
        print(f"智能投研助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式"""
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        print("=== 智能投研助手 - 终端模式 ===\n")
        print("请输入您的研究需求，我将按照深思熟虑的流程为您分析。")
        print("示例输入：'请分析新能源汽车行业在中期时间框架下的投资机会'")
        print("输入 'quit' 退出\n")
        
        while True:
            try:
                # 获取用户输入
                query = input('用户问题: ')
                
                if query.lower() == 'quit':
                    print("感谢使用智能投研助手！")
                    break
                
                # 输入验证
                if not query:
                    print('用户问题不能为空！')
                    continue
                    
                # 构建消息
                messages.append({'role': 'user', 'content': query})

                print("正在进行分析...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('助手回复:', response)
                messages.extend(response)
                
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举典型投研问题
        chatbot_config = {
            'prompt.suggestions': [
                '请分析新能源汽车行业在中期时间框架下的投资机会',
                '帮我研究人工智能技术在短期内的投资前景',
                '分析医疗健康行业在长期投资周期中的发展机会',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）
