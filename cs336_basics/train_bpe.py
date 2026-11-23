"""
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
"""

from __future__ import annotations
import os  #调用操作系统
import regex  ##正则表达式
import time  #调用计时器
from collections import Counter  #调用计数器(键不存在的键自动得到默认值0)
from typing import Union  #union表示联合类型,即多种类型之一

##GPT-2预分词模式
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """按所有特殊 token 把整段文本切成若干独立片段.
    Args:
        text: 读入的完整语料文本.
        special_tokens: 特殊 token 字符串列表.
    Returns:
        不含任何特殊 token 的文本片段列表.
    """
    chunks = [text]
    for special in special_tokens:
        next_chunks: list[str] = []
        for chunk in chunks:
            # 用普通的 str.split 按特殊 token 切分;
            # (原实现用 re.split + re.escape 防"|"被理解为"或",
            #  这里不用正则,天然不存在转义问题)
            next_chunks.extend(chunk.split(special))
        chunks = next_chunks
    return [chunk for chunk in chunks if chunk]
    # if chunk:非空才加入(对应原实现"非空才加入"的过滤)


def _collect_word_freqs(
    text: str, special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:
    """预分词并统计词频(对应原实现的 pre_tokenize_document,但直接操作原始字节).

    同一个词出现多次只需在后续合并循环里处理一次(加权即可),
    这是在大语料上也能跑得动的关键优化。

    与原实现的关键区别:原实现把每个字节经 bytes_to_unicode_map 映射成
    "可见字符"再训练;本实现直接在原始字节上训练。因为 bytes_to_unicode
    映射不保序(如字节 32→'Ġ' U+0120,字节 33→'!' U+0021),在映射后的
    字符上比较 pair 大小,结果与在原始字节上比较不一致,会导致平局时
    选出的 pair 与参考实现不同、merges 对不上。

    Args:
        text: 完整语料文本.
        special_tokens: 特殊 token 列表.

    Returns:
        Counter,键是「单字节 bytes 组成的元组」(一个词的字节序列),值是出现次数.
    """
    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    for chunk in _split_on_special_tokens(text, special_tokens):
        # 按照Unicode模式,用GPT-2的正则表达式把文本切分成token
        for match in regex.finditer(GPT2_SPLIT_PATTERN, chunk):
            # 把token编码成字节序列
            # 注意:tuple(bytes_obj) 得到的是 int 元组,必须逐个包成单字节 bytes
            # 词表示为「单字节 bytes 元组」,如 "low" -> (b"l", b"o", b"w")
            word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            word_freqs[word] += 1
    return word_freqs


class _BPEState:
    """BPE 训练状态(对应原实现的 BPEIndex 类,但用「增量计数 + 倒排索引」替代堆).

    维护:词表内的词、pair 频次、pair -> 词下标的倒排索引。
    每次合并只增量更新「受影响词」的 pair 计数,而不是全量重扫,
    保证 corpus.en 上远低于 1.5 秒的时限(test_train_bpe_speed)。
    """

    def __init__(self, word_freqs: Counter[tuple[bytes, ...]]) -> None:
        # words[i] 是第 i 种去重后的词(字节列表),freqs[i] 是它的出现次数
        self.words: list[list[bytes]] = []
        self.freqs: list[int] = []
        # 统计每个相邻字节对的出现次数,键是字节对元组,
        # 值是加权频次(出现次数 × 词频)
        self.pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        # 统计每个相邻字节对出现在哪些词里(倒排索引):
        # 键是字节对元组,值是词下标集合,合并时快速定位受影响词
        self.pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}

        # 统计频次和位置
        for word, freq in word_freqs.items():
            idx = len(self.words)
            self.words.append(list(word))
            self.freqs.append(freq)
            for first, second in zip(word, word[1:]):
                pair = (first, second)
                self.pair_counts[pair] += freq
                self.pair_to_words.setdefault(pair, set()).add(idx)

    def best_pair(self) -> tuple[bytes, bytes] | None:
        """返回当前最高频的 pair(对应原实现的 get_most_frequent).

        频次相同时取字节序更大的那个(CS336 参考实现约定)。
        max 的 key 是 (count, pair):先比频次,再比 pair 本身。
        pair 是原始字节元组,直接比较即可——不要映射成 GPT-2 可见字符再比
        (映射不保序,会导致 merges 与参考实现不一致)。
        (原实现用 [-count, pair] 最小堆,平局时弹出的是字典序更小的 pair,
         与参考实现相反,这是原代码 merges 对不上的核心原因。)

        Returns:
            最高频 pair;没有可合并的 pair 时返回 None.
        """
        if not self.pair_counts:
            return None
        return max(self.pair_counts.items(), key=lambda item: (item[1], item[0]))[0]

    def merge(self, pair: tuple[bytes, bytes], new_token: bytes) -> None:
        """把所有包含 pair 的词里的该 pair 合并为 new_token,并增量维护索引

        合并前:word = [..., word[pos-1], pair[0], pair[1], word[pos+2], ...]
                             ↑pos-1      ↑pos     ↑pos+1   ↑pos+2
        合并后:word = [..., word[pos-1], new_token, word[pos+2], ...]
                             ↑pos-1      ↑pos       ↑pos+1

        Args:
            pair: 要合并的相邻字节对.
            new_token: 合并后的新 token(pair[0] + pair[1])
        """
        affected = self.pair_to_words.get(pair)
        if not affected:
            return
        # 必须拷贝一份再遍历:循环体里会往 pair 自己的索引集合里 discard,
        # 直接遍历会报 "Set changed size during iteration"
        for idx in list(affected):
            old_word = self.words[idx]
            freq = self.freqs[idx]

            # 1) 撤销旧词的全部 pair 贡献
            # (对应原实现的 _update_pair_count(old_pair, -1):
            #  负责维护"每个pair出现几次",-1删除)
            for first, second in zip(old_word, old_word[1:]):
                old_pair = (first, second)
                self.pair_counts[old_pair] -= freq
                if self.pair_counts[old_pair] <= 0:
                    # 计数清零的 pair 直接删掉,best_pair 就不会再看到它
                    del self.pair_counts[old_pair]
                word_set = self.pair_to_words.get(old_pair)
                if word_set is not None:
                    word_set.discard(idx)

            # 2) 左到右贪心合并该词中的 pair(与参考实现一致)
            new_word: list[bytes] = []
            i = 0
            while i < len(old_word):
                if (
                    i < len(old_word) - 1
                    and old_word[i] == pair[0]
                    and old_word[i + 1] == pair[1]
                ):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            self.words[idx] = new_word

            # 3) 登记新词的全部 pair 贡献
            # (对应原实现的 _update_pair_count(new_pair, +1) 和 _add_position:
            #  +1更新,并添加新pair的位置信息到索引中)
            for first, second in zip(new_word, new_word[1:]):
                new_pair = (first, second)
                self.pair_counts[new_pair] += freq
                self.pair_to_words.setdefault(new_pair, set()).add(idx)


def run_train_bpe(
    input_path: Union[str, os.PathLike],  # 文件路径
    vocab_size: int,  # 词表目标数
    special_tokens: list[str],  # 特殊token列表
    **kwargs,  # 收集所有未显式声明的关键字参数(兼容 adapters.py 透传)
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """主训练函数:输出词表(原始+合并)和合并规则
    Args:
        input_path: 训练语料文件路径(UTF-8 文本).
        vocab_size: 目标词表大小(含 256 个字节 token 和特殊 token).
        special_tokens: 特殊 token 列表,只加入词表、不参与合并.
        **kwargs: 忽略的额外关键字参数.
    Returns:
        (vocab, merges):
            vocab  —— {token_id: token_bytes},id 0..255 是单字节,
                      256 起依次是特殊 token 和合并得到的新 token;
            merges —— 按创建顺序排列的合并规则 [(token1_bytes, token2_bytes), ...].
    """
    base_vocab_size = 256 + len(special_tokens)
    if vocab_size < base_vocab_size:
        raise ValueError(
            f"vocab_size({vocab_size}) 至少要等于 256 + len(special_tokens)"
            f" = {base_vocab_size}"
        )
        #raise ValueError表示提前报错提醒

    ## 1.加载语料
    # 用文本模式读取:Python 的通用换行会把 \r\n、\r 统一归一化成 \n。
    # 这一步是必须的——语料文件是 CRLF 行尾,官方参考 merges/vocab 和
    # test_train_bpe_special_tokens 的快照都是在归一化后的文本上生成的,
    # 若用二进制读取保留 \r,会训练出 b'\r\n'、b' \r' 之类官方没有的 token。
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    ## 2.预分词并统计词频(先按特殊token分割,再用GPT-2正则切词)
    word_freqs = _collect_word_freqs(text, special_tokens)

    ## 3.初始化词汇表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    ## 4.添加特殊token到词表(保持传入顺序,id 从 256 起)
    for special in special_tokens:
        # 把特殊token编码为utf-8字节,更新词表
        vocab[len(vocab)] = special.encode("utf-8")

    ## 5.BPE训练主循环
    merges: list[tuple[bytes, bytes]] = []  # 存放要合并的字节对
    state = _BPEState(word_freqs)  # 构建BPE索引

    while len(vocab) < vocab_size:
        # 记录目标词表规模和当前词表的差距:词表数量满 vocab_size 即停
        best_pair = state.best_pair()
        if best_pair is None:
            # 没有更多有效的字节对可供合并,提前结束训练
            break

        # 创建新token:把pair的两个字节串拼接起来
        new_token = best_pair[0] + best_pair[1]

        # 合并 + 扩表(按创建顺序记录)
        merges.append(best_pair)  # 合并字节对列表扩表
        vocab[len(vocab)] = new_token  # 词表字典扩表,id 顺序分配
        state.merge(best_pair, new_token)

    return vocab, merges  # 返回词表和合并字节对


## 主函数
if __name__ == "__main__":
    # 手动冒烟测试:直接跑 fixtures 里的小语料
    from pathlib import Path  ##调用路径

    # 依次尝试几个可能的 fixtures 位置(包内 -> 仓库根的 tests/ 下)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "fixtures",  # cs336_basics/fixtures(若把语料放进了包内)
        here.parent / "tests" / "fixtures",  # 仓库自带的 tests/fixtures
    ]
    corpus = next((c / "corpus.en" for c in candidates if (c / "corpus.en").exists()), None)
    if corpus is not None:
        start_time = time.time()
        vocab, merges = run_train_bpe(
            input_path=corpus,
            vocab_size=500,
            special_tokens=["<|endoftext|>"],  # <|endoftext|>(文档结束)
        )
        # print的f-string表达式:自动转换为str
        print(f"训练完成! 耗时: {time.time() - start_time:.2f}秒")
        print(f"词汇表大小: {len(vocab):,}")  # :,指的是用逗号做千位分隔
        print(f"合并操作数: {len(merges):,}")
        print(f"前 10 条合并规则: {merges[:10]}")
    else:
        print("未找到 fixtures/corpus.en,请通过 run_train_bpe() 传入语料路径使用。")
