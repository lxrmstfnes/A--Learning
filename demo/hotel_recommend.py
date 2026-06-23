#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店推荐 Demo
=============

基于 N 元分词(N-gram) + TF-IDF 向量化 + 余弦相似度的内容推荐系统。

用法:
    python hotel_recommend.py                          # 交互式输入文本描述
    python hotel_recommend.py "靠近派克市场，带游泳池"   # 单次文本描述推荐
    python hotel_recommend.py -t "luxury waterfront"   # 同上, 使用 -t 参数

核心思路:
    1. 将每家酒店的名称、地址、描述合并为一段文本
    2. 使用 N 元分词将文本切分为特征(如 "downtown seattle" 这样的二元组)
    3. 用 TF-IDF 将文本转为数值向量
    4. 用户输入文本描述后, 计算查询向量与所有酒店向量的余弦相似度
    5. 返回相似度最高的 Top-K 家酒店

依赖库: pandas, scikit-learn, numpy
"""

import argparse
import re
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 全局配置
# =============================================================================

# CSV 数据文件路径(与脚本同目录下的 Seattle_Hotels.csv)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Seattle_Hotels.csv")

# N 元分词范围: (1, 2) 表示同时使用 1-gram(单词) 和 2-gram(相邻两个词的组合)
# 例如 "downtown seattle hotel" 会生成:
#   1-gram: "downtown", "seattle", "hotel"
#   2-gram: "downtown seattle", "seattle hotel"
NGRAM_RANGE = (1, 2)

# 默认推荐数量
DEFAULT_TOP_K = 5

# 中文关键词 -> 英文扩展词(酒店语料为英文, 映射后可提升中文描述匹配效果)
CN_TO_EN_KEYWORDS = {
    "派克市场": "pike place market",
    "太空针": "space needle",
    "太空针塔": "space needle",
    "市中心": "downtown",
    "市区": "downtown",
    "豪华": "luxury",
    "高端": "luxury upscale",
    "经济": "budget affordable",
    "便宜": "budget affordable",
    "预算": "budget",
    "游泳池": "pool swimming",
    "泳池": "pool",
    "健身": "fitness gym workout",
    "健身房": "fitness center gym",
    "水疗": "spa",
    "温泉": "spa",
    "海滨": "waterfront bay",
    "水景": "waterfront view water",
    "湖景": "lake union view",
    "山景": "mountain view",
    "家庭": "family friendly",
    "亲子": "family friendly",
    "套房": "suite",
    "厨房": "kitchen",
    "早餐": "breakfast",
    "餐厅": "restaurant dining",
    "酒吧": "bar lounge",
    "艺术": "art museum gallery",
    "音乐": "music vinyl",
    "精品": "boutique",
    "商务": "business conference meeting",
    "会议": "conference meeting event",
    "步行": "walking distance walk",
    "附近": "near close nearby",
    "机场": "airport",
    "火车站": "train light rail",
    "购物": "shopping mall",
    "景点": "attraction landmark",
    "安静": "quiet peaceful",
    "景观": "view scenic",
    "免费": "complimentary free",
    "wifi": "wifi internet wireless",
    "无线网络": "wifi wireless internet",
}


# =============================================================================
# 文本预处理
# =============================================================================

def clean_text(text: str) -> str:
    """
    清洗原始文本, 去除噪声字符, 统一为小写.

    参数:
        text: 原始文本字符串

    返回:
        清洗后的文本字符串

    处理步骤:
        1. 处理空值(NaN / None)
        2. 转为小写(英文文本统一大小写, 避免 "Seattle" 和 "seattle" 被当作不同词)
        3. 将特殊符号替换为空格(保留字母和数字)
        4. 合并多余空白
    """
    # 空值保护: pandas 读取 CSV 时缺失字段可能是 NaN(float 类型)
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text).lower()

    # 保留英文字母、数字、中文及空格, 其余字符替换为空格
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)

    # 将连续多个空格压缩为一个空格, 并去除首尾空白
    text = re.sub(r"\s+", " ", text).strip()

    return text


def expand_query_with_keywords(query: str) -> str:
    """
    将用户文本描述中的中文关键词扩展为英文, 便于与英文酒店语料匹配.

    参数:
        query: 用户输入的原始文本描述

    返回:
        扩展后的查询文本(保留原文并追加英文关键词)
    """
    query = clean_text(query)
    if not query:
        return ""

    expansions = []
    for cn, en in CN_TO_EN_KEYWORDS.items():
        if cn in query:
            expansions.append(en)

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def build_corpus(row: pd.Series) -> str:
    """
    将酒店的一行数据(名称 + 地址 + 描述)合并为一段完整语料.

    参数:
        row: pandas Series, 包含 name、address、desc 字段

    返回:
        合并并清洗后的文本, 用于后续 N 元分词和向量化

    说明:
        合并多个字段可以让推荐同时匹配:
        - 酒店名称(如 "Hilton Garden Seattle")
        - 地理位置(如 "Pike Place Market")
        - 设施与特色描述(如 "fitness center", "waterfront view")
    """
    name = clean_text(row.get("name", ""))
    address = clean_text(row.get("address", ""))
    desc = clean_text(row.get("desc", ""))

    # 用空格连接三个字段, 形成一条完整的文档
    return " ".join(part for part in [name, address, desc] if part)


# =============================================================================
# 推荐引擎
# =============================================================================

class HotelRecommender:
    """
    酒店推荐引擎

    使用 TF-IDF + N 元分词构建酒店文档向量,
    通过余弦相似度为用户查询匹配最相关的酒店.
    """

    def __init__(self, csv_path: str = DATA_FILE, ngram_range: Tuple[int, int] = NGRAM_RANGE):
        """
        初始化推荐引擎, 加载数据并构建向量空间.

        参数:
            csv_path:  酒店 CSV 文件路径
            ngram_range: N 元分词范围, 如 (1, 2) 表示 1-gram 到 2-gram
        """
        self.csv_path = csv_path
        self.ngram_range = ngram_range

        # 存储原始酒店数据(DataFrame)
        self.hotels_df: pd.DataFrame = pd.DataFrame()

        # 每家酒店合并后的语料列表
        self.corpus: List[str] = []

        # TF-IDF 向量化器(fit 之后保存词表和 IDF 权重)
        self.vectorizer: TfidfVectorizer = None

        # 所有酒店的 TF-IDF 特征矩阵, 形状为 (酒店数量, 特征数量)
        self.hotel_matrix = None

        # 加载数据并训练向量化器
        self._load_and_vectorize()

    def _load_and_vectorize(self) -> None:
        """
        从 CSV 加载酒店数据, 构建语料库, 并用 TF-IDF + N 元分词向量化.

        TF-IDF 简介:
            - TF (Term Frequency): 词在当前文档中出现的频率
            - IDF (Inverse Document Frequency): 词的逆文档频率, 常见词权重低, 稀有词权重高
            - TF-IDF = TF * IDF, 综合衡量一个词对某篇文档的重要程度

        N 元分词的作用:
            - 1-gram 捕获单个关键词, 如 "pool", "spa"
            - 2-gram 捕获短语语义, 如 "pike place", "space needle"
            - 组合使用可提升推荐对短语查询的匹配能力
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"找不到数据文件: {self.csv_path}")

        # 读取 CSV
        # 数据文件可能包含非 UTF-8 字符(如 Windows 特殊空格), 使用 latin-1 兼容读取
        self.hotels_df = pd.read_csv(self.csv_path, encoding="latin-1")

        # 校验必要字段是否存在
        required_cols = {"name", "address", "desc"}
        missing = required_cols - set(self.hotels_df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要列: {missing}, 当前列: {list(self.hotels_df.columns)}")

        # 为每家酒店构建语料
        self.corpus = [build_corpus(row) for _, row in self.hotels_df.iterrows()]

        # 创建 TF-IDF 向量化器
        # max_features=10000: 最多保留 10000 个特征, 防止特征维度过高
        # stop_words='english': 过滤英文停用词(the, is, at 等无实际语义的词)
        # ngram_range: 指定 N 元分词范围
        # sublinear_tf=True: 使用 1 + log(tf) 代替原始 tf, 降低高频词的主导作用
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=10000,
            stop_words="english",
            sublinear_tf=True,
        )

        # fit_transform: 先在全部语料上学习词表和 IDF, 再转换为矩阵
        self.hotel_matrix = self.vectorizer.fit_transform(self.corpus)

        print(f"[初始化完成] 共加载 {len(self.hotels_df)} 家酒店")
        print(f"[向量化] N 元分词范围: {self.ngram_range[0]}-gram ~ {self.ngram_range[1]}-gram")
        print(f"[向量化] 特征维度: {self.hotel_matrix.shape[1]}")

    def recommend(self, description: str, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        """
        根据用户文本描述推荐最相似的酒店.

        参数:
            description: 用户输入的文本描述, 如 "靠近派克市场, 带游泳池的豪华酒店"
            top_k:       返回推荐酒店的数量

        返回:
            包含推荐结果的 DataFrame, 按相似度从高到低排序

        算法流程:
            1. 清洗并扩展用户文本描述(中文关键词映射为英文)
            2. 用同一个 vectorizer 将查询转为 TF-IDF 向量
            3. 计算查询向量与所有酒店向量的余弦相似度
            4. 取相似度最高的 top_k 家酒店
        """
        description = clean_text(description)
        if not description:
            raise ValueError("文本描述不能为空, 请描述您想要的酒店位置、设施或风格.")

        # 扩展中文关键词, 提升与英文语料的匹配能力
        search_query = expand_query_with_keywords(description)

        # 将查询文本转为 TF-IDF 向量
        # 注意使用 transform 而非 fit_transform, 保持与酒店矩阵同一特征空间
        query_vector = self.vectorizer.transform([search_query])

        # 计算余弦相似度
        # 余弦相似度公式: cos(theta) = (A dot B) / (|A| * |B|)
        # 取值范围 [-1, 1], 在 TF-IDF 非负向量中实际为 [0, 1]
        # 值越接近 1 表示两篇文档越相似
        similarities = cosine_similarity(query_vector, self.hotel_matrix).flatten()

        # 获取相似度最高的 top_k 个索引
        # argsort 升序排列, 取最后 k 个再反转得到降序
        top_indices = similarities.argsort()[::-1][:top_k]

        # 组装推荐结果
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            hotel = self.hotels_df.iloc[idx]
            desc_text = str(hotel["desc"])
            desc_summary = desc_text[:120] + "..." if len(desc_text) > 120 else desc_text
            results.append({
                "排名": rank,
                "相似度": round(float(similarities[idx]), 4),
                "酒店名称": hotel["name"],
                "地址": hotel["address"],
                "描述摘要": desc_summary,
            })

        return pd.DataFrame(results)

    def explain_match(self, query: str, hotel_index: int, top_features: int = 8) -> List[str]:
        """
        解释某家酒店与查询匹配的关键特征词(可选的调试/解释功能).

        参数:
            query:       用户查询
            hotel_index: 酒店在数据集中的索引(0-based)
            top_features: 展示的关键特征数量

        返回:
            与查询最相关的 N 元特征词列表
        """
        query = clean_text(query)
        query_vector = self.vectorizer.transform([query])
        hotel_vector = self.hotel_matrix[hotel_index]

        # 逐元素相乘: 两个向量在同一特征维度上都有值时, 乘积越大说明该特征对匹配贡献越大
        feature_scores = (query_vector.multiply(hotel_vector)).toarray().flatten()
        feature_names = self.vectorizer.get_feature_names_out()

        # 取得分最高的特征索引
        top_idx = feature_scores.argsort()[::-1][:top_features]
        return [feature_names[i] for i in top_idx if feature_scores[i] > 0]


# =============================================================================
# 演示与交互入口
# =============================================================================

def print_recommendations(results: pd.DataFrame, description: str) -> None:
    """
    格式化打印推荐结果.

    参数:
        results:     recommend() 返回的 DataFrame
        description: 用户输入的原始文本描述
    """
    print("\n" + "=" * 70)
    print(f"您的描述: {description}")
    print("=" * 70)

    for _, row in results.iterrows():
        print(f"\n【第 {row['排名']} 名】相似度: {row['相似度']}")
        print(f"  酒店: {row['酒店名称']}")
        print(f"  地址: {row['地址']}")
        print(f"  简介: {row['描述摘要']}")

    print("\n" + "=" * 70)


def recommend_from_description(
    recommender: HotelRecommender,
    description: str,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """
    根据单条文本描述执行推荐并打印结果.

    参数:
        recommender:  已初始化的推荐引擎实例
        description:  用户输入的文本描述
        top_k:        推荐数量
    """
    results = recommender.recommend(description, top_k=top_k)
    print_recommendations(results, description)


def run_interactive(recommender: HotelRecommender, top_k: int = DEFAULT_TOP_K) -> None:
    """
    交互式推荐模式: 用户输入文本描述, 实时返回推荐结果.

    参数:
        recommender: 已初始化的推荐引擎实例
        top_k:       每次推荐返回的酒店数量
    """
    print("\n>>> 请输入文本描述进行酒店推荐(输入 quit 或 exit 退出) <<<\n")
    print("示例描述:")
    print('  - "靠近派克市场, 步行可达, 带餐厅"')
    print('  - "豪华水景酒店, 有游泳池和水疗"')
    print('  - "family friendly suite near Space Needle with breakfast"\n')

    while True:
        try:
            description = input("请描述您想要的酒店 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if description.lower() in ("quit", "exit", "q", "退出"):
            print("再见!")
            break

        if not description:
            print("描述不能为空, 请重新输入.\n")
            continue

        try:
            recommend_from_description(recommender, description, top_k=top_k)
        except ValueError as e:
            print(f"错误: {e}\n")


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="基于文本描述的西雅图酒店推荐系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            '  python hotel_recommend.py\n'
            '  python hotel_recommend.py -t "靠近太空针, 带健身房"\n'
            '  python hotel_recommend.py "luxury waterfront spa pool"\n'
        ),
    )
    parser.add_argument(
        "description",
        nargs="*",
        help="酒店需求文本描述(可省略, 省略后进入交互模式)",
    )
    parser.add_argument(
        "-t", "--text",
        dest="text",
        help="酒店需求文本描述",
    )
    parser.add_argument(
        "-k", "--top",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"推荐酒店数量(默认 {DEFAULT_TOP_K})",
    )
    return parser.parse_args()


def main():
    """
    程序主入口.

    用法:
        python hotel_recommend.py                              # 交互式输入文本描述
        python hotel_recommend.py "靠近派克市场, 带游泳池"      # 单次文本描述推荐
        python hotel_recommend.py -t "luxury waterfront spa"   # 使用 -t 指定描述
        python hotel_recommend.py -k 3 "budget hotel downtown" # 指定推荐数量
    """
    args = parse_args()

    print("=" * 70)
    print("  西雅图酒店推荐系统  |  文本描述 + TF-IDF 相似度匹配")
    print("=" * 70)

    recommender = HotelRecommender()

    # 优先使用 -t/--text, 否则使用 positional 参数
    description = (args.text or " ".join(args.description)).strip()

    if description:
        recommend_from_description(recommender, description, top_k=args.top)
    else:
        run_interactive(recommender, top_k=args.top)


if __name__ == "__main__":
    main()
