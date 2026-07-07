"""
Encoder-Only Transformer 实现（PyTorch）

架构参考论文 "Attention Is All You Need"，仅保留 Encoder 部分，
适用于文本分类、序列表示学习等任务（类似 BERT 的编码器结构）。

用法:
    python transformer.py --device cuda   # GPU 训练与推理
    python transformer.py --device cpu    # CPU 训练与推理
    python transformer.py                 # 自动选择可用设备
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 设备工具
# ---------------------------------------------------------------------------

def get_device(device: Optional[str] = None) -> torch.device:
    """
    解析并返回计算设备。

    Args:
        device: 指定 "cpu" / "cuda" / "mps"，为 None 时自动选择。
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，已回退到 CPU")
        return torch.device("cpu")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("警告: MPS 不可用，已回退到 CPU")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# 位置编码（正弦 / 余弦，与原始 Transformer 论文一致）
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """为词嵌入注入位置信息，使模型感知 token 在序列中的顺序。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe 形状: (1, max_len, d_model)，广播到 batch 维度
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 不同频率的分母项，维度为 d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # 不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 多头自注意力
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    缩放点积多头注意力（Scaled Dot-Product Multi-Head Attention）。

    将 d_model 拆成 num_heads 个头，各自独立计算注意力后再拼接。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, num_heads, seq, d_k)"""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq, d_k) -> (batch, seq, d_model)"""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)，
                  True 表示该位置需要被屏蔽（设为 -inf）
        """
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        # 注意力分数: (batch, heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# 前馈网络（FFN）
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """两层全连接 + ReLU，中间维度通常为 d_model 的 4 倍。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


# ---------------------------------------------------------------------------
# 单个 Encoder 层
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Encoder 层 = 多头自注意力 + 残差 + LayerNorm + FFN + 残差 + LayerNorm
    （Post-LN 结构，与原始论文一致）
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 子层 1: 自注意力
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层 2: 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ---------------------------------------------------------------------------
# Encoder-Only Transformer（完整模型）
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    仅含 Encoder 的 Transformer。

    典型用途:
        - 序列分类: 取 [CLS] 位置或 mean pooling 后接分类头
        - 序列表示: 直接输出每个 token 的上下文向量
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_len: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # 词嵌入 + 缩放（论文中 embedding 乘以 sqrt(d_model)）
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # 分类头：对序列做 mean pooling 后线性映射到类别数
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化线性层权重，embedding 使用正态分布。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        生成 padding mask，屏蔽 pad token 的注意力。

        Args:
            src: (batch, seq_len) token id 序列
        Returns:
            (batch, 1, 1, seq_len)，pad 位置为 True
        """
        return (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码输入序列，返回每个位置的上下文表示。

        Args:
            src:      (batch, seq_len)
            src_mask: 可选的注意力 mask
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if src_mask is None:
            src_mask = self._make_padding_mask(src)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：编码 + mean pooling + 分类。

        Args:
            src: (batch, seq_len)
        Returns:
            logits: (batch, num_classes)
        """
        encoded = self.encode(src, src_mask)

        # 对非 pad 位置做 mean pooling，得到句子级表示
        pad_mask = (src != self.pad_idx).unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (encoded * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1.0)

        return self.classifier(pooled)

    @torch.no_grad()
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """推理接口，返回预测类别索引。"""
        self.eval()
        logits = self.forward(src)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# 演示：简单训练 + 推理
# ---------------------------------------------------------------------------

def _build_demo_data(
    batch_size: int = 8,
    seq_len: int = 16,
    vocab_size: int = 100,
    num_classes: int = 3,
    device: torch.device = torch.device("cpu"),
):
    """构造随机演示数据，便于快速验证模型能否正常训练。"""
    src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return src, labels


def train_demo(model: TransformerEncoder, device: torch.device, epochs: int = 5) -> None:
    """在随机数据上跑若干 epoch，验证训练流程。"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- 开始训练 (device={device}) ---")
    for epoch in range(1, epochs + 1):
        src, labels = _build_demo_data(device=device)
        optimizer.zero_grad()
        logits = model(src)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        print(f"Epoch {epoch:02d} | loss={loss.item():.4f} | acc={acc:.2%}")


def infer_demo(model: TransformerEncoder, device: torch.device) -> None:
    """推理演示：对一批随机序列输出预测类别。"""
    model.eval()
    src, _ = _build_demo_data(batch_size=4, device=device)
    preds = model.predict(src)
    print(f"\n--- 推理结果 (device={device}) ---")
    print(f"输入形状: {tuple(src.shape)}")
    print(f"预测类别: {preds.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encoder-Only Transformer 演示")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="指定计算设备；不填则自动选择",
    )
    parser.add_argument("--epochs", type=int, default=5, help="演示训练轮数")
    parser.add_argument("--d-model", type=int, default=128, help="模型隐藏维度")
    parser.add_argument("--num-heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=2, help="Encoder 层数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"使用设备: {device}")

    # 超参数（演示规模较小，CPU 也能快速跑通）
    vocab_size = 100
    num_classes = 3

    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_model * 4,
        max_len=64,
        num_classes=num_classes,
        dropout=0.1,
        pad_idx=0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    train_demo(model, device, epochs=args.epochs)
    infer_demo(model, device)


if __name__ == "__main__":
    main()
