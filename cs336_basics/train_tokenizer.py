from __future__ import annotations
import json
import regex
from collections.abc import Iterable, Iterator
from typing import List, Set, Tuple
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    整个编码解码连：
    encode：文本 → UTF-8字节 → 字节级BPE切分 → id序列
    decode：id序列 → 查vocab得字节块 → 拼接 → UTF-8解码 → 文本
    decode(encode(text))=text
    """
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens or []
        #存储合并字典的优先级：pair是合并对，i是序号
        self.merges_priority_map={pair:i for i,pair in enumerate(self.merges)}
        #将bytes转化为token id。
        self.bytes_to_id={byte:i for i,byte in self.vocab.items()}
        #题目要求：把不在词表里的特殊token追加进词表末尾
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)    # 新id=当前词表大小（即末尾位置）
                self.bytes_to_id[tok_bytes] = new_id
                self.vocab[new_id] = tok_bytes    #vocab是{id: bytes}，也要同步加

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从序列化的词表和合并列表构造Tokenizer"""
        import json  # 也可以放到文件头部 import
        # 与 tests/common.py 的 gpt2_bytes_to_unicode 一致的映射：
        # 字节 -> 可见字符；这里反着用，得到 可见字符 -> 字节
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        
        # byte_decoder: {可见字符: 原始字节(0~255)}
        byte_decoder = {chr(c): b for b, c in zip(bs, cs)}

        # 1.读词表：{id(int): bytes}
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(idx): bytes(byte_decoder[tok] for tok in token_str)
            for token_str, idx in raw_vocab.items()
        }

        # 2.读合并列表：[ (bytes, bytes), ... ]
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()          # 去掉行尾换行符
                if not cleaned:                   # 跳过空行
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:               # 跳过非标准行（有的文件尾有版本注释行）
                    continue
                t1, t2 = parts
                merges.append((
                    bytes(byte_decoder[c] for c in t1),
                    bytes(byte_decoder[c] for c in t2),
                ))

        # 3.构造
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _get_bpe_merges(self,piece:bytes)->List[bytes]:
        """ 输入字节段，输出一个合并后的字节列表
            Args: piece:bytes
            ->字节段转成单字节列表parts
            ->循环：
                记录所有可合并对；若没有，则循环该结束。
                维护new_parts列表，每轮只更新字典序最小（最先创建）的可合并pair
                把new_parts赋值更新给parts
            Returns: parts:List[bytes]
        """
        #将字节段piece转换为单字节列表
        parts=[bytes([b]) for b in piece]

        #循环合并字节列表，直接拼接起来
        while len(parts)>1:
            #记录所有合并对
            pairs=set()
            for i in range(len(parts)-1):
                pair=(parts[i],parts[i+1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
            
            #这一轮没有可合并的合并对，说明循环可以结束了
            if not pairs: break

            #找到最佳合并对
            #min函数：在pairs集合的元素中，找到对应的字典中序号最小的，返回最小元素
            best_pair=min(pairs,key=lambda pair:self.merges_priority_map[pair])

            #加入最佳合并对
            new_parts=[]
            i=0
            while i<len(parts):
                if i<len(parts)-1 and (parts[i],parts[i+1])==best_pair:
                    new_parts.append(parts[i]+parts[i+1])
                    i+=2
                else:
                    new_parts.append(parts[i])
                    i+=1
            parts=new_parts

        #经过循环后，得到最终合并字节列表
        return parts

    def encode(self,text:str)->List[int]:
        """输入一段str文本，输出经过BPE合并过后的token id列表"""
        """
            Args: text:str
            ->特殊token分割成List[str]
            ->每个str再预分词成List[str]
            ->每个str再转成utf-8bytes
            ->BPE合并，并把合并后的bytes转token id
            Returns: final_ids:List[str]
        """
        if not text: return []
        #按照长度降序给特殊字符排序，避免短token抢先匹配长token的前缀
        sorted_special_tokens=sorted(self.special_tokens,key=len,reverse=True)
        #把每个token用regex.escape转义后，用 | 拼成一个正则"或"模式
        special_token_pattern='|'.join(map(regex.escape,sorted_special_tokens))

        # 1.按照特殊token分割text，分割成chunks列表
        if self.special_tokens:
            chunks=regex.split(f'({special_token_pattern})',text)
        else:
            chunks=[text]
        
        # 2.对chunks列表中的每个chunk字符串，进行BPE合并化，并转成token id
        final_ids=[]
        for chunk in chunks:
            if not chunk: continue
            if chunk in self.special_tokens:
            #对于特殊字符chunk，直接将其从字符转为utf-8
            #再用self.bytes_to_id()函数转化为token id
                final_ids.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
            #对于正常chunk
                for word in regex.findall(PAT,chunk):
                #PAT是预定义的GPT-2预分词正则
                #先预分词chunk
                    if not word: continue
                    #先把字符转为utf-8字节，再BPE合并化
                    merged_pieces=self._get_bpe_merges(word.encode('utf-8'))

                    for piece in merged_pieces:
                        #bytes转token id
                        final_ids.append(self.bytes_to_id[piece])

        return final_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """encode的流式版本：充分利用可迭代对象(Iterable)的性质，边读边处理"""
        """适合大文本：可以惰性产出，不用一次性全部生成"""
        """减少内存占用：一次只占单条文本的token数"""
        # 字符串缓冲：存上一块末尾“可能是某个特殊token前缀”的片段
        buffer=""
        for text in iterable:
            buffer += text
            #计算能安全flush的字符数：从末尾往前，只要剩余部分
            #还可能是某个特殊token的前缀，就不能flush到它之前
            safe_len = len(buffer)
            for tok in self.special_tokens:
                #若buffer 的某个后缀恰是tok的前缀（且不是完整tok），
                #则该后缀必须保留，最多只能flush到它之前
                for k in range(1, len(tok)):
                    prefix = tok[:k]
                    if buffer.endswith(prefix):
                        safe_len = min(safe_len, len(buffer) - len(prefix))
            #flush掉[0,safe_len)这一段
            if safe_len > 0:
                yield from self.encode(buffer[:safe_len])
                buffer=buffer[safe_len:]
            #若safe_len==0，说明整段都是特殊token的前缀，留待下一块

        #流结束：flush剩下的buffer
        if buffer:
            yield from self.encode(buffer)


    def decode(self,ids:List[int])->str:
        """token_id序列转换为整块bytes，再转换为字符文本"""
        #合并解码的原因：字节级BPE无法考虑utf-8字符边界问题
        all_bytes=b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode("utf-8",errors="replace")
