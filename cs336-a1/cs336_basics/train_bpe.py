import regex
from collections import Counter

# 预分词切分，就按照正则规则切，这段看求不懂，语法也不知道咋写
def split(text: str, special_tokens: list[str]) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    

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
    blocks = split(text, special_tokens)


    
    return vocab, merges