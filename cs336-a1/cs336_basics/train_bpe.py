# input_path: 
# vocab_size: 
# special_tokens: 
# 返回 vocab, merges
# vocab: token id -> bytes(切割出来的字节串)
# merges: 合并操作记录，按合并顺序
def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return {}