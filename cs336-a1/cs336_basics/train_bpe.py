import regex
from collections import Counter

# 预分词切分，反正是按照固定规则切，正则规则看球不懂，regex 用不来，语法不知道咋写。
# 算了这两个函数当黑箱用。
def split_by_special(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(regex.escape(tok) for tok in special_tokens)
    return regex.split(pattern, text)
def split_by_regex(blocks: list[str]) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens: list[str] = []
    for block in blocks:
        pretokens.extend(regex.findall(PAT, block))
    return pretokens

# input_path: 纯文本
# vocab_size: 最终词表大小
# special_tokens: 不能合并的
# 返回 vocab, merges
# vocab: token id -> bytes(切割出来的字节串)
# merges: 合并操作记录，按合并顺序
def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256 + len(special_tokens)
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    for w in special_tokens:
        vocab[len(vocab)] = w.encode("utf-8")

    # 预分词
    blocks = split_by_special(text, special_tokens)
    pretokens = split_by_regex(blocks)

    # 原文本太长了，不能每次合并都去 query 一次
    cnt = Counter()
    for s in pretokens:
        # Counter 要求元素必须可以哈希，所以用 tuple 替代 list，对每个 pretoken 计数 +1
        cnt[tuple(bytes([b]) for b in s.encode())] += 1
    
    return vocab, merges