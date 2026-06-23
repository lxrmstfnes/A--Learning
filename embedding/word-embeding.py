#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec 词向量训练脚本
======================

使用 Gensim 对《三国演义》文本进行 Word Embedding 训练。

流程:
    1. 读取 embedding 目录下的 txt 语料
    2. 使用 jieba 对中文进行分词
    3. 用 Gensim Word2Vec 在分词结果上训练词向量
    4. 展示词向量示例并保存模型

依赖: gensim, jieba

用法:
    python word-embeding.py                    # 训练模型并进入交互查询
    python word-embeding.py -i                 # 加载已有模型, 直接进入交互查询
    python word-embeding.py --train-only       # 仅训练, 不进入交互
    python word-embeding.py --epochs 20        # 自定义训练参数
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import jieba
import numpy as np
from gensim.models import Word2Vec


# =============================================================================
# 路径与默认超参数
# =============================================================================

# 语料文件(与本脚本同目录)
CORPUS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Romance of the Three Kingdoms.txt",
)

# 训练完成后模型保存路径
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "three_kingdoms_word2vec.model",
)

# 词向量维度: 每个词用多少个数字表示(常见 100~300)
DEFAULT_VECTOR_SIZE = 150

# 窗口大小: 预测上下文时左右各看几个词(类比任务宜稍大)
DEFAULT_WINDOW = 10

# 最小词频: 出现次数低于此值的词会被丢弃
DEFAULT_MIN_COUNT = 3

# 训练轮数: 语料较小时需要更多轮次才能收敛
DEFAULT_EPOCHS = 40

# 并行线程数
DEFAULT_WORKERS = 4

# 高频词 subsampling 阈值(越大越 aggressively 丢弃高频词, 减轻「荆州/不能」等主导)
DEFAULT_SAMPLE = 1e-3

# 负采样数(Skip-gram 中每个正样本配对的负样本数)
DEFAULT_NEGATIVE = 15

# 三国演义主要人物名(用于分词词典 & 过滤被错误粘连的 token)
CHARACTER_NAMES = {
    "刘备", "玄德", "关羽", "云长", "张飞", "翼德", "曹操", "孙权",
    "诸葛亮", "孔明", "周瑜", "吕布", "袁绍", "马超", "黄忠", "赵云",
    "张辽", "司马懿", "姜维", "魏延", "庞统", "徐庶", "貂蝉", "董卓",
}


def _init_jieba_dict() -> None:
    """将人物名注入 jieba 词典, 避免「玄德答」「乃云长」等错误切分."""
    for name in CHARACTER_NAMES:
        jieba.add_word(name, freq=10000)


_init_jieba_dict()

# 中文停用词 + 三国演义语料中的高频功能词
# 这些词语义弱, 容易在 most_similar 中「霸榜」, 训练和展示时均需过滤
CHINESE_STOPWORDS = {
    "不能", "不可", "不可", "如此", "于是", "今日", "如何", "大喜", "忽报",
    "心中", "军士", "背后", "赶来", "嘱付", "遣使", "因此", "却说", "次日",
    "忽然", "且说", "只见", "一齐", "一齐", "大喜", "大惊", "商议", "引兵",
    "探马", "回报", "传令", "左右", "引着", "上马", "下寨", "探听", "分付",
    "主公", "荆州", "天下", "孔明曰", "玄德曰", "关公曰", "权曰", "操曰",
    "陛下", "寡人", "群臣", "某", "汝", "吾", "尔", "矣", "乎", "也",
    "这", "那", "一个", "什么", "不是", "就是", "已经", "因为", "所以",
    "如果", "可以", "没有", "这个", "他们", "我们", "你们", "自己",
}


# =============================================================================
# 文本预处理
# =============================================================================

def read_lines(corpus_path: str) -> List[str]:
    """
    读取语料文件, 按行返回非空文本.

    参数:
        corpus_path: txt 文件路径

    返回:
        去除空白行后的文本行列表
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"找不到语料文件: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"语料文件为空: {corpus_path}")

    return lines


def is_valid_token(token: str) -> bool:
    """
    判断分词结果是否保留.

    过滤掉:
        - 长度过短的词(单字符噪声)
        - 纯标点 / 纯数字
        - 停用词(不能/不可/荆州 等高频功能词)
        - 对话标记词(如 孔明曰、玄德曰)
    """
    if len(token) < 2:
        return False
    if re.fullmatch(r"[\W\d_]+", token):
        return False
    if token in CHINESE_STOPWORDS:
        return False
    if token.endswith("曰"):
        return False
    # 过滤 jieba 误切: 「玄德答」「云长领」= 人名 + 1~2 字, 非独立语义词
    for name in CHARACTER_NAMES:
        if token.startswith(name) and token != name and len(token) <= len(name) + 2:
            return False
    return True


def tokenize_line(line: str) -> List[str]:
    """
    对单行中文文本进行 jieba 分词并清洗.

    参数:
        line: 原始文本行

    返回:
        分词后的词列表, 如 ["玄德", "与", "关羽", "张飞", "结为", "兄弟"]
    """
    # jieba 返回生成器, 转为 list 后过滤无效 token
    tokens = jieba.lcut(line)
    return [t.strip() for t in tokens if is_valid_token(t.strip())]


def build_sentences(lines: List[str]) -> List[List[str]]:
    """
    将全部文本行转为 Word2Vec 需要的「句子」格式.

    Word2Vec 的输入不是整段字符串, 而是:
        [["词1", "词2", ...], ["词A", "词B", ...], ...]
    每一行文本对应一个句子(一个词列表).
    """
    sentences = []
    for line in lines:
        tokens = tokenize_line(line)
        if tokens:
            sentences.append(tokens)
    return sentences


# =============================================================================
# Word2Vec 训练
# =============================================================================

def train_word2vec(
    sentences: List[List[str]],
    vector_size: int = DEFAULT_VECTOR_SIZE,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    workers: int = DEFAULT_WORKERS,
    sample: float = DEFAULT_SAMPLE,
    negative: int = DEFAULT_NEGATIVE,
) -> Word2Vec:
    """
    使用 Gensim Word2Vec 训练词向量.

    参数:
        sentences:   分词后的句子列表
        vector_size: 词向量维度
        window:      上下文窗口大小
        min_count:   最小词频阈值
        epochs:      训练轮数
        workers:     并行线程数
        sample:      高频词 subsampling 阈值
        negative:    负采样数量

    返回:
        训练完成的 Word2Vec 模型

    说明:
        - sg=1 (Skip-gram): 更适合语义类比, 如 king-man+woman≈queen
        - sg=0 (CBOW):      训练更快, 但类比效果通常较差
        - sample:           降低「不能/荆州」等超高频词权重, 避免它们霸榜
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,                      # Skip-gram: 类比任务效果更好
        epochs=epochs,
        sample=sample,             # 高频词 subsampling
        negative=negative,         # 负采样
        ns_exponent=0.75,
    )
    return model


# =============================================================================
# 结果展示
# =============================================================================

def show_word_vector(model: Word2Vec, word: str) -> None:
    """
    打印单个词的向量(展示前 10 维).
    """
    if word not in model.wv:
        print(f"  词 '{word}' 不在词表中(可能词频过低被过滤)")
        return

    vector = model.wv[word]
    preview = ", ".join(f"{v:.4f}" for v in vector[:10])
    print(f"  '{word}' 向量(前10维/共{len(vector)}维): [{preview}, ...]")


def show_similar_words(model: Word2Vec, word: str, topn: int = 8) -> None:
    """
    打印与给定词语义最接近的 topn 个词(过滤停用词).
    """
    if word not in model.wv:
        print(f"  词 '{word}' 不在词表中, 跳过相似词查询")
        return

    print(f"  与 '{word}' 最相似的 {topn} 个词:")
    for similar_word, score in filtered_most_similar(model, word=word, topn=topn):
        print(f"    {similar_word:<8}  相似度: {score:.4f}")


def demo_analogies(model: Word2Vec) -> None:
    """
    演示向量类比, 类似英文 king - man + woman ≈ queen.

    三国演义中可成立的类比通常是:
        - 字号关系: 关羽≈云长, 张飞≈翼德, 刘备≈玄德
        - 阵营/关系: 关羽-云长+翼德≈张飞 (同一阵营的兄弟关系)
    """
    analogies = [
        (["云长"], ["关羽"], "云长 - 关羽 + ?  (期望: 玄德/翼德等字号关系)"),
        (["翼德"], ["张飞"], "翼德 - 张飞 + ?  (期望: 云长/玄德等)"),
        (["玄德"], ["刘备"], "玄德 - 刘备 + ?  (期望: 云长/翼德等)"),
        (["云长", "翼德"], ["关羽"], "云长 + 翼德 - 关羽 ≈ ?  (期望: 张飞)"),
        (["翼德", "云长"], ["张飞"], "翼德 + 云长 - 张飞 ≈ ?  (期望: 关羽)"),
    ]

    print("\n" + "=" * 60)
    print("【向量类比演示】  (类似 king - man + woman ≈ queen)")
    print("=" * 60)
    for positive, negative, desc in analogies:
        missing = check_words_in_vocab(model, positive + negative)
        if missing:
            print(f"\n  {desc}")
            print(f"    跳过: 词表缺少 {missing}")
            continue
        print(f"\n  {desc}")
        results = filtered_most_similar(
            model, positive=positive, negative=negative, topn=3,
            exclude=set(positive) | set(negative),
        )
        for rank, (w, score) in enumerate(results, start=1):
            print(f"    {rank}. {w}  (相似度: {score:.4f})")


def demo_model(model: Word2Vec) -> None:
    """
    用《三国演义》中的常见人物/词语演示训练效果.
    """
    demo_words = ["刘备", "关羽", "张飞", "曹操", "诸葛亮", "孙权"]

    print("\n" + "=" * 60)
    print("【词向量示例】")
    print("=" * 60)
    for word in demo_words[:3]:
        show_word_vector(model, word)
        print()

    print("=" * 60)
    print("【语义相似词】")
    print("=" * 60)
    for word in demo_words:
        show_similar_words(model, word, topn=5)
        print()

    demo_analogies(model)


def save_model(model: Word2Vec, model_path: str) -> None:
    """保存训练好的 Word2Vec 模型到磁盘."""
    model.save(model_path)
    print(f"\n[保存完成] 模型已写入: {model_path}")


def load_model(model_path: str) -> Word2Vec:
    """
    从磁盘加载已训练的 Word2Vec 模型.

    参数:
        model_path: .model 文件路径

    返回:
        加载完成的 Word2Vec 模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型文件: {model_path}\n"
            f"请先运行 python word-embeding.py --train-only 进行训练."
        )
    model = Word2Vec.load(model_path)
    print(f"[加载完成] 词表大小: {len(model.wv)} 个词")
    return model


# =============================================================================
# 词向量查询与运算
# =============================================================================

# 匹配向量运算表达式, 如 "曹操+刘备-张飞" 或 "曹操+刘备-张飞=x"
VECTOR_EXPR_PATTERN = re.compile(
    r"^[\s\u4e00-\u9fff\w]+(?:[\s]*[+-][\s]*[\u4e00-\u9fff\w]+)+$"
)


def check_words_in_vocab(model: Word2Vec, words: List[str]) -> List[str]:
    """
    检查词是否都在词表中, 返回缺失的词列表.
    """
    return [word for word in words if word not in model.wv]


def format_vector_preview(vector: np.ndarray, preview_size: int = 10) -> str:
    """格式化向量预览字符串."""
    preview = ", ".join(f"{v:.4f}" for v in vector[:preview_size])
    return f"[{preview}, ...]  (共 {len(vector)} 维)"


def filtered_most_similar(
    model: Word2Vec,
    word: Optional[str] = None,
    positive: Optional[List[str]] = None,
    negative: Optional[List[str]] = None,
    topn: int = 5,
    exclude: Optional[set] = None,
) -> List[Tuple[str, float]]:
    """
    获取最相似词, 并过滤停用词 / 对话标记 / 参与运算的词.

    Gensim 原始 most_similar 容易被「不能/荆州/主公」等高频词霸榜;
    这里多取一些候选, 过滤后再返回 topn 个有语义的词.
    """
    exclude = exclude or set()
    if word:
        exclude.add(word)

    if word:
        candidates = model.wv.most_similar(word, topn=topn + 80)
    else:
        candidates = model.wv.most_similar(
            positive=positive or [],
            negative=negative or [],
            topn=topn + 80,
        )

    results: List[Tuple[str, float]] = []
    for candidate, score in candidates:
        if candidate in exclude:
            continue
        if candidate in CHINESE_STOPWORDS:
            continue
        if candidate.endswith("曰"):
            continue
        # 过滤人名粘连 token
        contaminated = any(
            candidate.startswith(n) and candidate != n and len(candidate) <= len(n) + 2
            for n in CHARACTER_NAMES
        )
        if contaminated:
            continue
        results.append((candidate, score))
        if len(results) >= topn:
            break
    return results


def lookup_word(model: Word2Vec, word: str, topn: int = 1) -> None:
    """
    查询单个词: 展示其向量, 并找出最相近的词.

    参数:
        model: Word2Vec 模型
        word:  待查询的词
        topn:  返回最相似词的数量
    """
    word = word.strip()
    missing = check_words_in_vocab(model, [word])
    if missing:
        print(f"  词 '{word}' 不在词表中(可能词频过低或未出现在语料中).")
        return

    vector = model.wv[word]
    print(f"\n  词: {word}")
    print(f"  向量(前10维): {format_vector_preview(vector)}")

    print(f"\n  与 '{word}' 最相近的词:")
    results = filtered_most_similar(model, word=word, topn=5)
    if not results:
        print("    (未找到合适的相似词)")
        return
    for similar_word, score in results[:1]:
        print(f"    → {similar_word}  (相似度: {score:.4f})")
    if len(results) > 1:
        print(f"  其他相近词:")
        for similar_word, score in results[1:]:
            print(f"    · {similar_word}  (相似度: {score:.4f})")


def parse_vector_expression(expr: str) -> Tuple[List[str], List[str]]:
    """
    解析向量运算表达式, 拆分为 positive(加项) 和 negative(减项).

    支持格式:
        曹操+刘备-张飞
        曹操 + 刘备 - 张飞
        曹操+刘备-张飞=x   (等号右侧忽略, x 代表待求结果)

    参数:
        expr: 用户输入的运算表达式

    返回:
        (positive_words, negative_words)
        例如 "曹操+刘备-张飞" → (["曹操", "刘备"], ["张飞"])
    """
    expr = expr.strip().replace(" ", "")
    if "=" in expr:
        expr = expr.split("=")[0].strip()

    tokens = re.findall(r"[\u4e00-\u9fff\w]+|[+-]", expr)
    if not tokens:
        raise ValueError("表达式为空.")

    positive: List[str] = []
    negative: List[str] = []
    current_sign = "+"

    for token in tokens:
        if token == "+":
            current_sign = "+"
        elif token == "-":
            current_sign = "-"
        elif current_sign == "+":
            positive.append(token)
        else:
            negative.append(token)

    if not positive and not negative:
        raise ValueError("表达式中没有有效的词.")

    return positive, negative


def compute_vector_expression(
    model: Word2Vec,
    expr: str,
    topn: int = 5,
) -> None:
    """
    执行词向量加减运算, 并输出最相近的结果词.

    原理:
        曹操 + 刘备 - 张飞 ≈ x
        即: vec(曹操) + vec(刘备) - vec(张飞) 所指向的方向,
        在词表中找到与之最接近的词 x.

    经典例子(英文): king - man + woman ≈ queen

    参数:
        model: Word2Vec 模型
        expr:  如 "曹操+刘备-张飞" 或 "曹操+刘备-张飞=x"
        topn:  返回最相近的 topn 个词
    """
    positive, negative = parse_vector_expression(expr)

    all_words = positive + negative
    missing = check_words_in_vocab(model, all_words)
    if missing:
        print(f"  以下词不在词表中: {', '.join(missing)}")
        return

    # 展示参与运算的词向量
    print("\n  【参与运算的词向量】")
    for word in all_words:
        sign = "+" if word in positive else "-"
        print(f"    {sign} {word}: {format_vector_preview(model.wv[word])}")

    # 计算结果向量: sum(positive) - sum(negative)
    result_vector = np.zeros(model.vector_size)
    for word in positive:
        result_vector += model.wv[word]
    for word in negative:
        result_vector -= model.wv[word]

    expr_display = expr.split("=")[0].strip()
    print(f"\n  【运算结果向量】  {expr_display} = x")
    print(f"    x 向量(前10维): {format_vector_preview(result_vector)}")

    # 在词表中找到与结果向量最相近的词(过滤停用词)
    print(f"\n  【最相近的词 x】(Top {topn}):")
    results = filtered_most_similar(
        model,
        positive=positive,
        negative=negative,
        topn=topn,
        exclude=set(positive) | set(negative),
    )
    for rank, (word, score) in enumerate(results, start=1):
        print(f"    {rank}. {word}  (相似度: {score:.4f})")


def is_vector_expression(text: str) -> bool:
    """判断输入是否为向量加减表达式(含 + 或 -)."""
    text = text.strip().replace(" ", "")
    if "=" in text:
        text = text.split("=")[0]
    return bool(VECTOR_EXPR_PATTERN.match(text))


def handle_query(model: Word2Vec, user_input: str) -> None:
    """
    根据用户输入分发到单词查询或向量运算.

    参数:
        model:      Word2Vec 模型
        user_input: 用户输入的文本
    """
    user_input = user_input.strip()
    if not user_input:
        print("  输入不能为空.")
        return

    print("\n" + "-" * 50)
    if is_vector_expression(user_input):
        compute_vector_expression(model, user_input)
    else:
        lookup_word(model, user_input, topn=1)
    print("-" * 50)


def run_interactive(model: Word2Vec) -> None:
    """
    交互式查询模式.

    支持两种输入:
        1. 单个词       → 查看向量 + 最相近的词
        2. 向量运算式   → 如 曹操+刘备-张飞=x
    """
    print("\n" + "=" * 60)
    print("  词向量交互查询  (输入 quit 或 exit 退出)")
    print("=" * 60)
    print("\n  用法示例:")
    print("    曹操              → 查看向量及最相近词")
    print("    翼德+云长-张飞     → 向量类比(期望≈关羽, 类似 king-man+woman≈queen)")
    print("    云长+翼德-关羽     → 向量类比(期望≈张飞)\n")

    while True:
        try:
            user_input = input("请输入 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if user_input.lower() in ("quit", "exit", "q", "退出"):
            print("再见!")
            break

        if not user_input:
            continue

        try:
            handle_query(model, user_input)
        except ValueError as e:
            print(f"  解析错误: {e}\n")


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="使用 Gensim Word2Vec 训练中文词向量")
    parser.add_argument(
        "--corpus",
        default=CORPUS_FILE,
        help="语料 txt 文件路径",
    )
    parser.add_argument(
        "--output",
        default=MODEL_FILE,
        help="模型保存路径",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=DEFAULT_VECTOR_SIZE,
        help=f"词向量维度(默认 {DEFAULT_VECTOR_SIZE})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"上下文窗口大小(默认 {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help=f"最小词频(默认 {DEFAULT_MIN_COUNT})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"训练轮数(默认 {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行线程数(默认 {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="加载已有模型, 直接进入交互查询(不重新训练)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="仅训练并保存模型, 不进入交互查询",
    )
    return parser.parse_args()


def train_pipeline(args: argparse.Namespace) -> Word2Vec:
    """执行完整的训练流程并保存模型."""
    print("=" * 60)
    print("  Word2Vec 词向量训练  |  Gensim + jieba")
    print("=" * 60)

    print(f"\n[1/4] 读取语料: {args.corpus}")
    lines = read_lines(args.corpus)
    print(f"      共 {len(lines)} 行")

    print("\n[2/4] jieba 分词中...")
    sentences = build_sentences(lines)
    token_count = sum(len(s) for s in sentences)
    print(f"      有效句子: {len(sentences)} 条, 总词数: {token_count}")

    print("\n[3/4] 训练 Word2Vec...")
    print(f"      参数: vector_size={args.vector_size}, window={args.window}, "
          f"min_count={args.min_count}, epochs={args.epochs}")
    model = train_word2vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
    )
    print(f"      词表大小: {len(model.wv)} 个词")

    print("\n[4/4] 训练完成, 展示结果...")
    demo_model(model)
    save_model(model, args.output)
    return model


def main() -> None:
    """
    主入口.

    模式:
        默认           → 训练模型 → 进入交互查询
        -i             → 加载已有模型 → 交互查询
        --train-only   → 仅训练, 不进入交互
    """
    args = parse_args()

    if args.interactive:
        model = load_model(args.output)
        run_interactive(model)
        return

    model = train_pipeline(args)

    if not args.train_only:
        run_interactive(model)


if __name__ == "__main__":
    main()
