# CS336 作业 1（基础）：构建一个 Transformer 语言模型

**版本 26.0.3**

CS336 课程组

2026 年春季学期

## 1 作业概览

在本次作业中，你将从头构建训练一个标准 Transformer 语言模型（language model，LM）所需的全部组件，并实际训练一些模型。

**你需要实现的内容**

1. Byte-pair encoding（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数与 AdamW 优化器（第 4 节）
4. 训练循环，并支持模型与优化器状态的序列化保存与加载（第 5 节）

**你需要运行的内容**

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer 语言模型。
4. 使用训练好的 Transformer 语言模型进行采样生成，并评估其 perplexity（困惑度）。
5. 在 OpenWebText 上训练模型，并将你取得的 perplexity 提交到排行榜。

**你可以使用的内容**

我们希望你对每一个组件都从头构建。特别地，除了以下内容之外，你**不能**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）¹
- `torch.optim.Optimizer` 基类

你可以使用任何其他的 PyTorch 定义。如果你想使用某个函数或类，但不确定是否被允许，欢迎到 Slack 上提问。


**代码长什么样**

本次作业的代码以及这份说明文档都可以在 GitHub 上找到：

`github.com/stanford-cs336/assignment1-basics`

请先 `git clone` 该仓库。如果之后有任何更新，我们会通知你，届时你 `git pull` 即可获取最新版本。

1. `cs336_basics/*`：这是你写代码的地方。注意这里现在没有任何代码——你可以完全从头自由发挥！
2. `adapters.py`：这里定义了一组你的代码必须提供的功能。对于每一项功能（例如 scaled dot-product attention），你只需通过调用你自己的代码来填好对应的实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的改动不应包含任何实质性的逻辑，它只是胶水代码。
3. `test_*.py`：这里包含了你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），它们会调用 `adapters.py` 中定义的钩子。**不要编辑测试文件。**

**如何提交**

要提交作业，运行 `make_submission.sh` 来构建一个提交用的 zip 文件。如果你有不想包含在提交 zip 里的大型数据文件或 checkpoint，记得把它们加到脚本的排除列表里。你需要向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题。请务必排版你的作答。
- `code.zip`：包含你写的全部代码。

要向排行榜提交，请向以下仓库提交一个 PR：

`github.com/stanford-cs336/assignment1-basics-leaderboard`

详细的提交说明见排行榜仓库中的 `README.md`。

**从哪里获取数据集**

本次作业将使用两个预处理好的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。这两个数据集都是单一的、大型的纯文本文件。

如果你是跟班上课的同学，下载数据集的说明可以在 compute guide 中找到。

如果你是自学，可以按照 `README.md` 里的命令来下载这些文件。

> **低资源提示：开篇**
>
> 在整个课程的作业讲义中，我们会给出一些建议，帮助你在 GPU 资源较少甚至没有的情况下完成作业的各个部分。例如，我们有时会建议你缩小数据集或模型规模，或者讲解如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色方框里看到这些"低资源提示"（就像这一个）。即使你是注册在籍、能使用课程机器的斯坦福学生，这些提示也能帮你更快迭代、节省时间，所以我们推荐你读一读！

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用助教提供的参考代码，我们可以在配备 36 GB 内存的 Apple M4 Max 芯片上训练出一个语言模型，生成相当流畅的文本——在 Metal GPU（MPS）上不到 5 分钟，用 CPU 则大约 30 分钟。如果这些名词对你来说还很陌生，别担心！你只需要知道：只要你的笔记本配置还算新，而且你的实现正确、高效，你就能训练出一个小的语言模型，以不错的流畅度生成简单的儿童故事。
>
> 在后文的作业中，我们会说明如果你是在 CPU 或 MPS 上运行，需要做哪些改动。

## 2 Byte-Pair Encoding（BPE）分词器

在作业的第一部分，我们将训练并实现一个字节级 byte-pair encoding（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们会把任意（Unicode）字符串表示为字节序列，并在这个字节序列上训练我们的 BPE 分词器。之后，我们会用这个分词器把文本（字符串）编码成 token（整数序列），用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它把字符映射为整数码点（code point）。截至 Unicode 17.0（2025 年 9 月发布），该标准定义了跨 172 种文字体系的 159,801 个字符。例如，字符 "s" 的码点是 115（通常写作 U+0073，其中 `U+` 是惯用前缀，`0073` 是 115 的十六进制表示），而字符 "牛" 的码点是 29275。在 Python 中，你可以用 `ord()` 函数把单个 Unicode 字符转换成它的整数表示，用 `chr()` 函数把一个整数 Unicode 码点转换成包含相应字符的字符串。

```
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**Problem (unicode1)：理解 Unicode（1 分）**

(a) `chr(0)` 返回的是哪个 Unicode 字符？

> **交付物**：一句话作答。

(b) 这个字符的字符串表示（`__repr__()`）与它的打印（print）表示有何不同？

> **交付物**：一句话作答。

(c) 当这个字符出现在文本中时，会发生什么？在 Python 解释器里试试下面这些，看看是否符合你的预期，可能会对你有帮助：

```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

> **交付物**：一句话作答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不现实的，因为词表会大得惊人（约 150K 项），而且非常稀疏（因为很多字符都极其罕见）。取而代之，我们会使用一种 Unicode 编码，它把 Unicode 字符转换成字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主流编码（超过 98% 的网页都使用它）。

要把一个 Unicode 字符串编码成 UTF-8，我们可以使用 Python 中的 `encode()` 函数。要访问 Python bytes 对象的底层字节值，我们可以遍历它（例如调用 `list()`）。最后，我们可以用 `decode()` 函数把一个 UTF-8 字节串解码成 Unicode 字符串。

```
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串的字节值（0 到 255 之间的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节并不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过把 Unicode 码点转换成字节序列（例如借助 UTF-8 编码），我们本质上就是把一个码点序列（21 位整数，有 159,801 个有效取值）转换成一个字节值序列（取值在 0 到 255 之间的整数）。长度为 256 的字节词表就好处理得多了。使用字节级分词时，我们不必担心 out-of-vocabulary（词表外）token 的问题，因为我们知道任何输入文本都可以表示为 0 到 255 之间的整数序列。

**Problem (unicode2)：Unicode 编码（3 分）**

(a) 有哪些理由让我们更倾向于在 UTF-8 编码的字节上训练分词器，而不是 UTF-16 或 UTF-32？对比一下这几种编码对不同输入字符串的输出，可能会对你有帮助。

> **交付物**：一到两句话作答。

(b) 考虑下面这个（错误的）函数，它的本意是把 UTF-8 字节串解码成 Unicode 字符串。为什么这个函数是错的？请给出一个会得到错误结果的输入字节串示例。

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

> **交付物**：一个会让 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，并用一句话解释该函数错在哪里。

(c) 给出一个无法解码为任何 Unicode 字符的两字节序列。

> **交付物**：一个示例，并用一句话解释。

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的 out-of-vocabulary 问题，但把文本切分成字节会产生极长的输入序列，从而拖慢模型训练。一个 10 个词的句子在词级语言模型里可能只有 10 个 token，但在字符级模型里却可能有 50 个甚至更多 token（取决于词的长度）。处理这些更长的序列，模型的每一步都需要更多计算。此外，在字节序列上做语言建模也很困难，因为更长的输入序列会在数据中制造长程依赖。

子词分词（subword tokenization）是词级分词器和字节级分词器之间的一个折中。注意，字节级分词器的词表有 256 项（字节值从 0 到 255）。子词分词器用更大的词表规模，来换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中频繁出现，那么为它在词表中分配一项，就能把这个 3-token 的序列压缩成单个 token。那么，我们该如何挑选要加入词表的子词单元呢？R. Sennrich et al. [3] 提出使用 byte-pair encoding（BPE；P. Gage [5]），这是一种压缩算法，它会迭代地把出现最频繁的字节对替换（"合并"）成一个新的、尚未使用的索引。注意，这个算法向词表中加入子词 token，是为了最大化对输入序列的压缩——如果一个词在输入文本中出现的次数足够多，它就会被表示成单个子词单元。

用 BPE 构建词表的子词分词器通常被称为 BPE 分词器。在本次作业中，我们会实现一个字节级 BPE 分词器，它的词表项是字节或者合并后的字节序列，这样既能处理好 out-of-vocabulary，又能保证输入序列长度可控，两全其美。构建 BPE 分词器词表的过程，被称为"训练" BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器的训练过程包含三个主要步骤。

**词表初始化**

分词器词表是从字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器，初始词表就是所有字节的集合。字节值共有 256 种可能，因此我们的初始词表大小为 256。

**预分词（pre-tokenization）**

有了词表之后，原则上你可以统计文本里哪些字节相邻出现，然后从出现最频繁的字节对开始合并。但这样做计算开销很大，因为我们每次合并都得完整地遍历一遍语料。此外，直接在语料上跨词合并字节，可能会产生仅标点不同的 token（例如 `dog!` 和 `dog.`），这些 token 会被分配完全不同的 token ID，尽管它们的语义很可能高度相似（因为它们只是标点不同）。

为了避免这个问题，我们对语料进行预分词。你可以把它理解为对语料做的一种粗粒度切分，帮助我们统计字符对出现的频率。例如，单词 `'text'` 可能是一个出现了 10 次的 pre-token。在这种情况下，当我们统计字符 `'t'` 和 `'e'` 相邻出现的次数时，就知道单词 `'text'` 里有 `'t'` 和 `'e'` 相邻，于是可以直接把它们的计数加 10，而不用遍历语料。由于我们训练的是字节级 BPE 模型，每个 pre-token 都被表示为一个 UTF-8 字节序列。

R. Sennrich et al. [3] 的原始 BPE 实现是通过简单地按空白切分（即 `s.split(" ")`）来做预分词。这种方法在基于 SentencePiece 的分词器里仍然可见（例如 Llama 1 和 2 的分词器）。

大多数现代分词器使用基于正则表达式的预分词器，这是从 GPT-2 沿袭下来的做法（A. Radford et al. [6]）。我们会使用原始正则的一个稍微规整些的版本，取自 `github.com/openai/tiktoken/pull/234/files`：

```
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

用这个预分词器交互式地切分一些文本，可以帮你更好地理解它的行为：

```
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过，在代码中使用它时，你应该用 `re.finditer`，这样在构建从 pre-token 到计数的映射时，就不必先把预分词后的词全部存下来。

**计算 BPE 合并**

现在我们已经把输入文本转换成了 pre-token，并把每个 pre-token 表示成了 UTF-8 字节序列，就可以计算 BPE 合并（也就是训练 BPE 分词器）了。概括地说，BPE 算法会迭代地统计每一对字节，找出出现频率最高的那一对（"A"，"B"），然后把这对最频繁的字节对的所有出现都合并掉，也就是替换成一个新的 token "AB"。这个新合并出来的 token 会被加入词表；因此，BPE 训练完成后的最终词表大小，等于初始词表大小（我们的情况是 256）加上训练过程中执行的 BPE 合并次数。为了提高 BPE 训练的效率，我们不考虑跨越 pre-token 边界的字节对。² 在计算合并时，如果多个字节对频率并列，我们就确定性地取字典序更大的那一对。例如，如果 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都是最高频率，我们就合并 ("BA", "A")：

```
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**特殊 token**

很多时候，一些字符串（例如 `<|endoftext|>`）被用来编码元数据（例如文档之间的边界）。在编码文本时，通常希望把某些字符串当作"特殊 token"处理，让它们永远不会被拆分成多个 token（即始终作为一个整体 token 保留）。例如，序列结束字符串 `<|endoftext|>` 应当始终作为单个 token（即单个整数 ID）保留，这样我们才知道何时停止从语言模型生成。这些特殊 token 必须加入词表，从而拥有一个对应的、固定的 token ID。R. Sennrich et al. [3] 的算法 1 给出了一个低效的 BPE 分词器训练实现（本质上就是遵循我们上面列出的步骤）。作为第一个练习，实现并测试这个函数可能会很有用，帮助你检验自己的理解。

**Example (bpe_example)：BPE 训练示例**

这里是一个取自 R. Sennrich et al. [3] 的示意性示例。考虑一个由以下文本构成的语料：

```
low low low low low lower lower widest widest widest newest newest newest newest newest newest
```

并且词表里有一个特殊 token `<|endoftext|>`。

**词表**

我们用特殊 token `<|endoftext|>` 和 256 个字节值来初始化词表。

**预分词**

为简单起见，也为了聚焦于合并过程，在这个示例里我们假设预分词只按空白切分。预分词并统计后，我们得到如下的频率表：

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

把它表示成 `dict[tuple[bytes, ...], int]` 会很方便，例如 `{(l,o,w): 5, …}`。注意，在 Python 中即便单个字节也是一个 bytes 对象。Python 中没有表示单个字节的 byte 类型，正如没有表示单个字符的 char 类型一样。

**合并**

我们首先看每一对相邻字节，并把它们出现的单词频率累加起来，得到 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。其中 `('e', 's')` 和 `('s', 't')` 并列，所以我们取字典序更大的那一对，即 `('s', 't')`。然后我们合并这些 pre-token，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

第二轮，我们发现 `(e, st)` 是最常见的对（计数为 9），合并后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。如此继续下去，最终得到的合并序列是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们只取 6 次合并，就得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，词表元素为 `[<|endoftext|>, [...256 个字节字符...], st, est, ow, low, west, ne]`。

用这个词表和这组合并，单词 `newest` 会被分词为 `[ne, west]`。

### 2.5 实验：BPE 分词器训练

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明见第 1 节。开始之前，我们建议你先浏览一下 TinyStories 数据集，对数据内容有个大致了解。

**并行化预分词**

你会发现一个主要的性能瓶颈在预分词这一步。你可以用内置库 `multiprocessing` 并行化你的代码来加速预分词。具体来说，我们建议在并行实现预分词时，把语料分块，同时确保分块边界落在特殊 token 的开头。你可以直接照搬下面链接里的起始代码来获取分块边界，然后用它把工作分发到各个进程：

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

这种分块方式始终是有效的，因为我们从不希望跨文档边界合并。就本次作业而言，你总是可以这样切分。不必担心出现"收到一个不包含 `<|endoftext|>` 的超大语料"这种边缘情况。

**预分词前移除特殊 token**

在用正则模式（使用 `re.finditer`）做预分词之前，你应该先从语料（如果你的实现是并行的，则从每个分块）中剔除所有特殊 token。务必按特殊 token 切分，这样就不会在它们所分隔的文本之间发生合并。例如，如果你的语料（或分块）形如 `[Doc 1]<|endoftext|>[Doc 2]`，你就应该按特殊 token `<|endoftext|>` 切分，并分别对 `[Doc 1]` 和 `[Doc 2]` 做预分词，这样就不会跨越文档边界合并。换句话说，特殊 token 在训练中定义了硬分割边界，但它们本身不应计入合并统计。这可以用 `re.split` 来实现，用 `"|".join(special_tokens)` 作为分隔符（要小心使用 `re.escape`，因为特殊 token 中可能出现 `|`）。测试 `test_train_bpe_special_tokens` 会检验这一点。

**优化合并步骤**

上面示意性示例中朴素的 BPE 训练实现很慢，因为每次合并都要遍历所有字节对来找出最频繁的那一对。然而，每次合并后，唯一会改变的字节对计数，是那些与刚合并的字节对重叠的部分。因此，可以通过为所有字节对的计数建立索引并增量式更新这些计数，来提升 BPE 训练速度，而不必显式地遍历每一对字节去统计频率。通过这种缓存手段，你能获得显著的加速。不过要注意，BPE 训练中的合并部分在 Python 里是无法并行化的。

> **低资源提示：性能分析**
>
> 你应该使用诸如 `cProfile` 或 `py-spy` 之类的性能分析工具，找出实现中的瓶颈，并集中精力优化它们。

> **低资源提示："缩小规模"**
>
> 与其一步到位地在完整 TinyStories 数据集上训练分词器，我们建议你先在一个小的数据子集——一个"调试数据集"——上训练。例如，你可以先在 TinyStories 的验证集上训练分词器，它只有 22K 篇文档，而不是 2.12M 篇。这体现了一个通用的策略：尽可能缩小规模来加速开发，例如使用更小的数据集、更小的模型规模等等。选择调试数据集或超参数配置的规模需要仔细考量：你要让调试集足够大，使其具备与完整配置相同的瓶颈（这样你做的优化才有普适性），但又不能大到要跑很久。

**Problem (train_bpe)：BPE 分词器训练（15 分）**

> **交付物**：写一个函数，给定一个输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应至少处理以下输入参数：

**输入**

- `input_path: str` —— 包含 BPE 分词器训练数据的文本文件路径。
- `vocab_size: int` —— 一个正整数，定义最终词表的最大规模（包括初始字节词表、合并产生的词表项以及任何特殊 token）。
- `special_tokens: list[str]` —— 要加入词表的字符串列表。在训练时，把它们当作硬边界，阻止跨其范围的合并，但在计算合并统计时不把它们计入。

你的 BPE 训练函数应返回生成的词表和合并列表：

**输出**

- `vocab: dict[int, bytes]` —— 分词器词表，从 int（词表中的 token ID）到 bytes（token 字节）的映射。
- `merges: list[tuple[bytes, bytes]]` —— 训练产生的 BPE 合并列表。列表中的每一项都是一个 bytes 元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 发生了合并。合并列表应按创建顺序排列。

要针对我们提供的测试来检验你的 BPE 训练函数，你需要先实现测试适配器 `[adapters.run_train_bpe]`。然后运行 `uv run pytest tests/test_train_bpe.py`，你的实现应能通过所有测试。

（可选，这可能是一项很大的时间投入）你可以用某种系统级语言来实现训练方法的关键部分，例如 C++（可考虑 `cppyy` 或 `nanobind`）或 Rust（使用 `PyO3`）。如果你这么做，请注意区分哪些操作需要拷贝、哪些可以直接从 Python 内存读取，并务必留下构建说明，或者确保只用 `pyproject.toml` 就能构建。另外请注意，GPT-2 的正则在大多数正则引擎中支持得并不好，在少数支持的引擎里又往往太慢。我们已确认 Oniguruma 速度尚可且支持负向前瞻，但 Python 的 `regex` 包甚至更快。

**Problem (train_bpe_tinystories)：在 TinyStories 上训练 BPE（2 分）**

(a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词表规模为 10,000。务必把 TinyStories 的 `<|endoftext|>` 特殊 token 加入词表。把生成的词表和合并列表序列化到磁盘以便后续查看。训练花了多少时间和内存？词表中最长的 token 是什么？这合理吗？

> **资源要求**：≤ 30 分钟（无需 GPU），≤ 30 GB 内存

> **提示**：利用预分词时的多进程，以及下面两个事实，你应该能在 2 分钟内完成 BPE 训练：
> (a) `<|endoftext|>` token 在数据文件中分隔文档。
> (b) `<|endoftext|>` token 在应用 BPE 合并之前作为特殊情况处理。

> **交付物**：一到两句话作答。

(b) 对你的代码做性能分析。分词器训练过程中哪一部分最耗时？

> **交付物**：一到两句话作答。

接下来，我们尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。和之前一样，我们建议你先浏览一下数据集，以便更好地理解其内容。

**Problem (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE（2 分）**

(a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词表规模为 32,000。把生成的词表和合并列表序列化到磁盘以便后续查看。词表中最长的 token 是什么？这合理吗？

> **资源要求**：≤ 12 小时（无需 GPU），≤ 100 GB 内存

> **交付物**：一到两句话作答。

(b) 对比你在 TinyStories 和 OpenWebText 上训练得到的分词器。

> **交付物**：一到两句话作答。

### 2.6 BPE 分词器：编码与解码

在作业的上一部分，我们实现了一个函数，用于在输入文本上训练 BPE 分词器，得到词表和 BPE 合并列表。现在，我们要实现一个 BPE 分词器，它加载给定的词表和合并列表，并用它们在文本和 token ID 之间进行编码和解码。

#### 2.6.1 编码文本

用 BPE 编码文本的过程，与我们训练 BPE 词表的过程是呼应的。主要有几个步骤。

**步骤 1：预分词。** 我们先对序列做预分词，并把每个 pre-token 表示成 UTF-8 字节序列，就像 BPE 训练时做的那样。我们会在每个 pre-token 内部把这些字节合并成词表元素，每个 pre-token 独立处理（不跨越 pre-token 边界合并）。

**步骤 2：应用合并。** 然后，我们取 BPE 训练时创建的词表元素合并序列，并按照它们创建的顺序，应用到我们的 pre-token 上。

**Example (bpe_encoding)：BPE 编码示例**

例如，假设我们的输入字符串是 `'the cat ate'`，词表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的合并列表是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，我们的预分词器会把这个字符串切分成 `['the', ' cat', ' ate']`。然后，我们逐个查看每个 pre-token 并应用 BPE 合并。

第一个 pre-token `'the'` 初始表示为 `[b't', b'h', b'e']`。查看我们的合并列表，找到第一个可用的合并是 `(b't', b'h')`，用它把 pre-token 变换成 `[b'th', b'e']`。然后，我们回到合并列表，找到下一个可用的合并是 `(b'th', b'e')`，把 pre-token 变换成 `[b'the']`。最后，再回看合并列表，发现没有更多可应用的合并了（因为整个 pre-token 已经合并成单个 token），于是 BPE 合并到此结束，对应的整数序列是 `[9]`。对其余的 pre-token 重复这个过程：pre-token `' cat'` 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`；最后一个 pre-token `' ate'` 应用合并后是 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，编码这个输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊 token**

你的分词器在编码文本时，应能正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考量**

假设我们想分词一个无法完整载入内存的大型文本文件。为了高效地分词这个大文件（或任何其他数据流），我们需要把它切分成可管理的分块，逐块处理，这样内存复杂度就是常数级，而不是随文本大小线性增长。这样做时，我们必须确保一个 token 不会跨越分块边界，否则得到的分词结果会和"把整个序列一次性载入内存做分词"的朴素方法不一致。

#### 2.6.2 解码文本

要把整数 token ID 序列解码回原始文本，我们只需查每个 ID 在词表中对应的条目（字节序列），把它们拼接起来，然后把字节解码成 Unicode 字符串。注意，输入的 ID 并不保证能映射到合法的 Unicode 字符串（因为用户可以输入任意的整数 ID 序列）。当输入的 token ID 不能产生合法 Unicode 字符串时，你应该用官方的 Unicode 替换字符 U+FFFD 替换那些损坏的字节。³ `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 会自动用替换标记替换损坏的数据。

**Problem (tokenizer)：实现分词器（15 分）**

> **交付物**：实现一个 `Tokenizer` 类，给定词表和合并列表，把文本编码成整数 ID，并把整数 ID 解码成文本。你的分词器还应支持用户提供的特殊 token（如果它们不在词表中，就把它们追加进去）。我们推荐以下接口：

```
def __init__(self, vocab, merges, special_tokens=None)
```

根据给定的词表、合并列表，以及（可选的）特殊 token 列表构造一个分词器。该函数应接受以下参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

类方法，从一个序列化的词表和合并列表（格式与你的 BPE 训练代码输出一致），以及（可选的）特殊 token 列表，构造并返回一个 `Tokenizer`。该方法应接受以下额外参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```
def encode(self, text: str) -> list[int]
```

把输入文本编码成 token ID 序列。

```
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定一个字符串可迭代对象（例如一个 Python 文件句柄），返回一个惰性地逐项产出 token ID 的生成器。这对于无法直接载入内存的大文件的高效分词是必需的。

```
def decode(self, ids: list[int]) -> str
```

把 token ID 序列解码成文本。

要针对我们提供的测试来检验你的 `Tokenizer`，你需要先实现测试适配器 `[adapters.get_tokenizer]`。然后运行 `uv run pytest tests/test_tokenizer.py`，你的实现应能通过所有测试。

### 2.7 实验

**Problem (tokenizer_experiments)：分词器实验（4 分）**

(a) 从 TinyStories 和 OpenWebText 中各采样 10 篇文档。用你之前训练好的 TinyStories 和 OpenWebText 分词器（词表规模分别为 10K 和 32K），把这些采样文档编码成整数 ID。每个分词器的压缩比（bytes/token）是多少？

> **交付物**：一到两句话作答。

(b) 如果你用 TinyStories 分词器来分词 OpenWebText 的采样，会发生什么？比较压缩比，和/或定性描述发生了什么。

> **交付物**：一到两句话作答。

(c) 估算你分词器的吞吐量（例如以 bytes/second 为单位）。分词整个 Pile 数据集（825GB 文本）需要多长时间？

> **交付物**：一到两句话作答。

(d) 用你的 TinyStories 和 OpenWebText 分词器，把各自的训练集和开发集编码成整数 token ID 序列。我们稍后会用它来训练语言模型。我们建议把 token ID 序列化成一个 dtype 为 `uint16` 的 NumPy 数组。为什么 `uint16` 是个合适的选择？

> **交付物**：一到两句话作答。

## 3 Transformer 语言模型架构

一个语言模型接收一批整数 token ID 序列作为输入（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回词表上的一个（分批的）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中预测的分布是针对每个输入 token 的"下一个词"。训练语言模型时，我们用这些下一个词的预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理阶段从语言模型生成文本时，我们取最后一个时间步（即序列中的最后一项）的预测下一个词分布来生成序列中的下一个 token（例如取概率最高的 token、从分布中采样等），把生成的 token 追加到输入序列，然后重复这个过程。

在作业的这一部分，你将从头构建这个 Transformer 语言模型。我们先对模型做一个高层描述，再逐步展开各个组件的细节。

### 3.1 Transformer LM

给定一个 token ID 序列，Transformer 语言模型先用输入 embedding 把 token ID 转换成稠密向量，让这些经过嵌入的 token 依次通过 `num_layers` 个 Transformer block，然后应用一个可学习的线性投影（"输出 embedding"或 "LM head"）来产生预测的下一个 token 的 logits。示意图见图 1。

*[图 1：我们 Transformer 语言模型的概览。形状为 (batch_size, seq_len, d_model) 的输入张量 → 若干 Transformer block → 线性层（输出 embedding）→ softmax → 输出概率]*

*[图 2：一个 pre-norm Transformer block。Norm → 带 RoPE 的因果多头自注意力 → Add → Norm → 逐位置前馈网络 → Add]*

**Token Embeddings**

在最开始的一步，Transformer 把（分批的）token ID 序列嵌入成一串包含 token 身份信息的向量（图 1 中的红色块）。更具体地说，给定一个 token ID 序列，Transformer 语言模型用一个 token embedding 层来产生一串向量。每个 embedding 层接收一个形状为 `(batch_size, sequence_length)` 的整数张量，产出一串形状为 `(batch_size, sequence_length, d_model)` 的向量。

**Pre-norm Transformer Block**

嵌入之后，激活值会经过若干结构相同的神经网络层处理。一个标准的 decoder-only Transformer 语言模型由 `num_layers` 个相同的层组成（通常称为 Transformer "block"）。每个 Transformer block 接收形状为 `(batch_size, sequence_length, d_model)` 的输入，返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个 block 都会在序列维度上聚合信息（通过自注意力），并对其进行非线性变换（通过前馈层）。

经过 `num_layers` 个 Transformer block 之后，我们取最终的激活值，把它们转成词表上的一个分布。我们将实现 "pre-norm" 的 Transformer block（详见第 3.4 节），它额外要求在最后一个 Transformer block 之后使用 layer normalization（详见下文），以确保其输出被恰当地缩放。在这次归一化之后，我们用一个标准的可学习线性变换，把 Transformer block 的输出转换成预测的下一个 token 的 logits（参见，例如 A. Radford et al. [7] 的公式 2）。

### 3.2 说明：批处理、einsum 与高效计算

在整个 Transformer 中，我们会反复对许多"批式"输入执行相同的计算。这里有几个例子：

- **批中的元素**：我们对批中的每个元素都应用相同的 Transformer 前向操作。
- **序列长度**：像 RMSNorm 和前馈网络这样的"逐位置（position-wise）"操作，在序列的每个位置上执行完全相同的运算。
- **注意力头**：注意力操作在"多头"注意力中跨注意力头批式执行。

如果能有一种顺手的方式来执行这些操作，既充分利用 GPU，又易于阅读和理解，会很有帮助。许多 PyTorch 操作可以在张量开头接收额外的"批式"维度，并在这些维度上高效地重复/广播该操作。例如，假设我们在做一个逐位置的、批式的操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" D，想与一个形状为 `(d_model, d_model)` 的矩阵 A 做批式的向量-矩阵乘法。这时，`D @ A` 会执行批式矩阵乘法——这是 PyTorch 中一个高效的原语，其中 `(batch_size, sequence_length)` 这两个维度被批量处理。

正因如此，一个有用的做法是：假设你的函数可能被传入额外的批式维度，并把这些维度保持在 PyTorch 形状的开头。为了让张量能够以这种方式批处理，它们可能需要进行很多步 `view`、`reshape` 和 `transpose` 来整形。这有点麻烦，而且常常会让人很难读懂代码在做什么、张量的形状是什么。

更顺手的选择是使用 `torch.einsum` 中的 einsum 记法，或者使用与框架无关的库，如 `einops` 或 `einx`。两个关键操作是 `einsum`（可以对输入张量的任意维度做张量收缩）和 `rearrange`（可以对任意维度做重排、拼接和拆分）。事实证明，机器学习中几乎所有的操作，都是某种"维度腾挪"与"张量收缩"的组合，偶尔夹带一个（通常是逐点的）非线性函数。这意味着，使用 einsum 记法时，你的很多代码会变得更具可读性、更灵活。

我们强烈建议你学习并在课程中使用 einsum 记法。没有接触过 einsum 记法的同学应使用 `einops`（文档在这里），而已经熟悉 `einops` 的同学应学习更通用的 `einx`（在这里）。⁴ 这两个包都已经安装在我们提供的环境里了。

下面我们给出几个 einsum 记法用法的例子。这些是 `einops` 文档的补充，你应该先读文档。

**Example (einstein_example1)：用 einops.einsum 做批式矩阵乘法**

```python
import torch
from einops import rearrange, einsum

# 基本实现
Y = D @ A.T  # 很难看出输入输出形状是什么、各代表什么含义。
# D 和 A 可以是什么形状？其中有没有哪些会产生出乎意料的行为？

# einsum 是自文档化且健壮的
# D A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

# 或者，一个批式版本：D 可以有任意前导维度，但 A 的维度是固定的。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**Example (einstein_example2)：用 einops.rearrange 做广播操作**

我们有一批图像，希望根据某个缩放因子，为每张图像生成 10 个变暗的版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

# 整形再相乘
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

# 或者一步到位：
dimmed_images = einsum(
    images, dim_by, "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**Example (einstein_example3)：用 einops.rearrange 做像素混合**

假设我们有一批图像，表示为一个形状为 `(batch, height, width, channel)` 的张量，我们想对图像的所有像素做一个线性变换，但这个变换对每个通道是独立进行的。我们的线性变换表示为一个形状为 `(height * width, height * width)` 的矩阵 B。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

# 重排图像张量以跨所有像素混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改用 einops：

```python
height = width = 32
# rearrange 取代了笨拙的 torch view + transpose
channels_first = rearrange(
    channels_last, "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B, "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed, "batch channel (height width) -> batch height width channel",
    height=height, width=width,
)
```

或者，如果你想来点刺激的，用 `einx.dot`（einx 中 `einops.einsum` 的对应物）一步到位：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width,
)
```

上面的第一种实现，本可以在前后加上注释来说明输入输出形状，但那样笨拙且容易出 bug。用 einsum 记法，文档本身就是实现！einsum 记法既能处理任意的输入批式维度，还有一个关键好处——它是自文档化的：使用 einsum 记法的代码里，输入输出张量的相关形状一目了然。对于剩下的张量，你可以考虑使用 Tensor 类型提示，例如 `jaxtyping` 库（并非 JAX 专属）。

关于 einsum 记法的性能影响，我们会在作业 2 中进一步讨论，现在你只需要知道：它们几乎总是优于其他替代方案！

#### 3.2.1 数学记法与内存顺序

许多机器学习论文在记法中使用行向量，这产生的表示与 NumPy 和 PyTorch 默认使用的行主序（row-major）内存顺序相契合。用行向量时，线性变换写作

$$y = xW^\top \tag{1}$$

其中 $W \in \mathbb{R}^{d_{out}\times d_{in}}$ 为行主序，行向量 $x \in \mathbb{R}^{1\times d_{in}}$。注意，这样我们就可以通过增大 $x$ 的最外层维度来实现输入批处理，也就是说，我们可以把向量输入 $x$ 替换成矩阵输入 $X \in \mathbb{R}^{batch\times d_{in}}$。

在线性代数里，通常更常见的是用列向量，此时线性变换写作

$$y = Wx \tag{2}$$

其中 $W \in \mathbb{R}^{d_{out}\times d_{in}}$ 为行主序，列向量 $x \in \mathbb{R}^{d_{in}}$。在这种设定下要对输入做批处理，$x$ 的批维度就得放在最后，所以 $x$ 需要被替换成矩阵 $\tilde{X} \in \mathbb{R}^{d_{in}\times batch}$。

在本次作业中，数学记法我们大多使用列向量，因为数学上通常遵循这种记法。你要记住：如果你想使用普通的矩阵乘法记法，就必须像公式 1 中的行向量约定那样，对矩阵做转置后再相乘，因为 PyTorch 使用行主序内存顺序。如果你用 einsum 来做线性代数运算，只要把各轴标注正确，这就不是问题。顺带一提，值得注意的是，像 Matlab、Julia 和 Fortran 等其他语言/线性代数包都使用列主序内存顺序，即批维度放在最后；而 Python 及其相关包则采用了 C 标准的行主序。

### 3.3 基本构建块：Linear 与 Embedding 模块

#### 3.3.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化会导致诸如梯度消失或梯度爆炸等不良行为。pre-norm Transformer 对初始化异常地鲁棒，但初始化仍可能对训练速度和收敛产生显著影响。由于本次作业已经很长了，我们把细节留到作业 3，这里先给你一些在大多数情况下都能正常工作的近似初始化。现在请使用：

- **Linear 权重**：$\mathcal{N}(\mu=0, \sigma^2=\frac{2}{d_{in}+d_{out}})$，截断到 $[-3\sigma, 3\sigma]$。
- **Embedding**：$\mathcal{N}(\mu=0, \sigma^2=1)$，截断到 $[-3, 3]$。
- **RMSNorm**：$\mathbb{1}$

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

Linear 层是 Transformer 以及一般神经网络的基本构建块。首先，你要实现自己的 `Linear` 类，继承自 `torch.nn.Module`，执行线性变换：

$$y = Wx \tag{3}$$

注意，我们不加偏置项，这与大多数现代 LLM 的做法一致。

**Problem (linear)：实现 linear 模块（1 分）**

> **交付物**：实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，只是没有 bias 参数。我们推荐以下接口：

```
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造一个线性变换模块。该函数应接受以下参数：

- `in_features: int` —— 输入的最后一个维度。
- `out_features: int` —— 输出的最后一个维度。
- `device: torch.device | None = None` —— 存储参数的设备。
- `dtype: torch.dtype | None = None` —— 参数的数据类型。

```
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用线性变换。

务必做到：

- 继承 `nn.Module`；
- 调用父类构造函数；
- 构造并把你的参数存储为 $W$（而非 $W^\top$），将其放入 `nn.Parameter` 中；
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`。

初始化时，使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的 `Linear` 模块，实现测试适配器 `[adapters.run_linear]`。该适配器应把给定的权重加载进你的 `Linear` 模块，你可以用 `Module.load_state_dict` 来完成。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是一个 embedding 层，它把整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个自定义的 `Embedding` 类，继承自 `torch.nn.Module`（所以你不应使用 `nn.Embedding`）。它的 `forward` 方法应通过索引一个形状为 `(vocab_size, d_model)` 的 embedding 矩阵，为每个 token ID 选出对应的 embedding 向量，输入是形状为 `(batch_size, sequence_length)` 的 token ID 的 `torch.LongTensor`。

**Problem (embedding)：实现 embedding 模块（1 分）**

> **交付物**：实现继承自 `torch.nn.Module` 并执行 embedding 查表的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：

```
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造一个 embedding 模块。该函数应接受以下参数：

- `num_embeddings: int` —— 词表大小。
- `embedding_dim: int` —— embedding 向量的维度，即 $d_{model}$。
- `device: torch.device | None = None` —— 存储参数的设备。
- `dtype: torch.dtype | None = None` —— 参数的数据类型。

```
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

为给定的 token ID 查表得到 embedding 向量。

务必做到：

- 继承 `nn.Module`；
- 调用父类构造函数；
- 把 embedding 矩阵初始化为 `nn.Parameter`；
- 存储 embedding 矩阵时，让 `d_model` 作为最后一个维度；
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`。

同样，初始化使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的实现，实现测试适配器 `[adapters.run_embedding]`。然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer Block

每个 Transformer block 有两个子层：一个多头自注意力机制，和一个逐位置前馈网络（A. Vaswani et al. [8] 第 3.1 节）。

在原始的 Transformer 论文中，模型在两个子层周围各使用一个残差连接，然后接 layer normalization。这种架构通常被称为 "post-norm" Transformer，因为 layer normalization 应用在子层的输出上。然而，多项工作发现，把 layer normalization 从每个子层的输出移到每个子层的输入（并在最后一个 Transformer block 之后再额外加一层 layer normalization），能提升 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——这种 "pre-norm" Transformer block 的直观表示见图 2。每个 Transformer block 子层的输出，随后通过残差连接加到子层的输入上（A. Vaswani et al. [8] 第 5.4 节）。对 pre-norm 的一种直觉理解是：从输入 embedding 到 Transformer 的最终输出，存在一条干净的、不经任何归一化的"残差流"，据称这能改善梯度流动。这种 pre-norm Transformer 如今已成为语言模型的标准（例如 GPT-3、LLaMA、PaLM 等），所以我们将实现这一变体。我们会逐一讲解 pre-norm Transformer block 的每个组件，并按顺序实现它们。

#### 3.4.1 Root Mean Square Layer Normalization（RMSNorm）

A. Vaswani et al. [8] 的原始 Transformer 实现使用 layer normalization [J. L. Ba et al., 2016] 来归一化激活值。遵循 H. Touvron et al. [12]，我们将使用 root mean square layer normalization（RMSNorm；B. Zhang et al. [13] 公式 4）来做 layer normalization。给定一个激活值向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 会对每个激活值 $a_i$ 做如下缩放：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i \tag{4}$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$。这里，$g_i$ 是一个可学习的 "gain"（增益）参数（总共有 `d_model` 个这样的参数），$\varepsilon$ 是一个超参数，通常固定为 `1e-5`。你应该把输入向上转型（upcast）为 `torch.float32`，以避免在平方输入时溢出。总体而言，你的 `forward` 方法应形如：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# 在这里写执行 RMSNorm 的代码
...
result = ...

# 以原始 dtype 返回结果
return result.to(in_dtype)
```

**Problem (rmsnorm)：Root Mean Square Layer Normalization（1 分）**

> **交付物**：把 RMSNorm 实现为一个 `torch.nn.Module`。我们推荐以下接口：

```
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 RMSNorm 模块。该函数应接受以下参数：

- `d_model: int` —— 模型的隐藏维度。
- `eps: float = 1e-5` —— 用于数值稳定性的 epsilon 值。
- `device: torch.device | None = None` —— 存储参数的设备。
- `dtype: torch.dtype | None = None` —— 参数的数据类型。

```
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，返回同形状的张量。

注意：记得在归一化之前把输入向上转型为 `torch.float32`（之后再向下转型回原始 dtype），如上所述。

要测试你的实现，实现测试适配器 `[adapters.run_rmsnorm]`。然后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

*[图 3：比较 SiLU（又称 Swish）和 ReLU 激活函数。SiLU: f(x) = x·σ(x)；Identity: f(x) = x；ReLU: f(x) = max(0, x)]*

在原始 Transformer 论文（A. Vaswani et al. [8] 第 3.3 节）中，Transformer 的前馈网络由两个线性变换组成，中间夹一个 ReLU 激活（$\text{ReLU}(x)=\max(0,x)$）。在那个原始架构中，前馈网络内层的维度通常是输入维度的 4 倍。

然而，现代语言模型相比这个原始设计，通常会做两处主要改动：使用另一种激活函数，并采用一种门控机制。具体来说，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 中采用的 "SwiGLU" 激活函数，它把 SiLU（常被称为 Swish）激活与一种称为 Gated Linear Unit（GLU）的门控机制结合了起来。我们也会省略线性层有时使用的偏置项，这与自 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来的大多数现代 LLM 一致。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$\text{SiLU}(x) = x\cdot\sigma(x) = \frac{x}{1+e^{-x}} \tag{5}$$

如图 3 所示，SiLU 激活函数与 ReLU 激活函数相似，但在零点处是平滑的。Gated Linear Unit（GLU）最初由 Y. N. Dauphin et al. [19] 定义为：一个经过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x)\odot W_2 x \tag{6}$$

其中 $\odot$ 表示逐元素乘法。Gated Linear Unit 被认为能"通过为梯度提供一条线性通路，同时保留非线性能力，来缓解深层架构中的梯度消失问题"。

把 SiLU/Swish 与 GLU 结合起来，就得到了 SwiGLU，我们将把它用于前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1 x)\odot W_3 x) \tag{7}$$

其中 $x\in\mathbb{R}^{d_{model}}$，$W_1, W_3\in\mathbb{R}^{d_{ff}\times d_{model}}$，$W_2\in\mathbb{R}^{d_{model}\times d_{ff}}$，且规范地取 $d_{ff}=\frac{8}{3}d_{model}$。在具体实现中，出于硬件效率，把它四舍五入到附近的 64 的倍数是没问题的。

N. Shazeer [20] 首先提出把 SiLU/Swish 激活与 GLU 结合，并做了实验，表明在语言建模任务上 SwiGLU 优于 ReLU 和 SiLU（无门控）等基线。在作业的后半部分，你会比较 SwiGLU 和 SiLU。虽然我们为这些组件提到了一些启发式论证（论文也提供了更多支持性证据），但保持经验主义的视角是好的——Shazeer 论文中有一句如今已相当出名的话：

> "对于这些架构为何似乎有效，我们不做任何解释；我们将其成功，如同其他一切一样，归因于神明的仁慈。"

**Problem (positionwise_feedforward)：实现逐位置前馈网络（2 分）**

> **交付物**：实现 SwiGLU 前馈网络，它由一个 SiLU 激活函数和一个 GLU 组成。

> 注意：在这种特定情况下，为了数值稳定性，你可以放心地在实现中使用 `torch.sigmoid`。

在你的实现中，应把 $d_{ff}$ 设为约 $\frac{8}{3}\times d_{model}$，同时确保前馈内层的维度是 64 的倍数，以便充分利用硬件。要针对我们提供的测试来检验你的实现，你需要实现测试适配器 `[adapters.run_swiglu]`。然后运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.4.3 相对位置嵌入

为了向模型注入位置信息，我们将实现 Rotary Position Embedding [J. Su et al., 2021]，通常称为 RoPE。对于 token 位置 $i$ 处的查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^{d}$，我们会应用一个成对旋转矩阵 $R_i$，得到 $q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}$。这里，$R_i$ 会把 embedding 元素对 $q^{(i)}_{2k-1:2k}$ 作为二维向量，旋转角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$（其中 $k\in\{1,\dots,d/2\}$，$\Theta$ 是某个常数）。因此，我们可以把 $R_i$ 看作一个大小为 $d\times d$ 的块对角矩阵，块为 $R^i_k$（$k\in\{1,\dots,\frac{d}{2}\}$），其中

$$R^i_k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix} \tag{8}$$

于是我们得到完整的旋转矩阵

$$R_i = \begin{pmatrix} R^i_1 & 0 & 0 & \cdots & 0 \\ 0 & R^i_2 & 0 & \cdots & 0 \\ 0 & 0 & R^i_3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R^i_{d/2} \end{pmatrix} \tag{9}$$

其中 0 代表 $2\times 2$ 的零矩阵。虽然可以构造完整的 $d\times d$ 矩阵，但一个好的实现应该利用这个矩阵的性质，以更高效的方式实现变换。由于我们只关心序列内 token 之间的相对旋转，我们可以跨层、跨不同的 batch 复用在 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 上计算出的值。如果你想优化，可以让所有层共享同一个 RoPE 模块，并在 init 时用 `self.register_buffer(persistent=False)` 创建一个预先计算好的 2D sin/cos 值 buffer，而不是用 `nn.Parameter`（因为我们不希望学习这些固定的余弦和正弦值）。我们对 $q^{(i)}$ 做的完全相同的旋转过程，也会对 $k^{(j)}$ 执行，旋转相应的 $R_j$。注意，这一层没有可学习参数。

**Problem (rope)：实现 RoPE（2 分）**

> **交付物**：实现一个类 `RotaryPositionalEmbedding`，对输入张量应用 RoPE。

推荐以下接口：

```
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE 模块，必要时创建 buffer。

- `theta: float` —— RoPE 的 Θ 值。
- `d_k: int` —— query 和 key 向量的维度。
- `max_seq_len: int` —— 输入的最大序列长度。
- `device: torch.device | None = None` —— 存储 buffer 的设备。

```
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，返回同形状的张量。注意，你应当容忍 $x$ 带任意数量的批维度。你应假设 token 位置是形状为 `(..., seq_len)` 的张量，指明 $x$ 沿序列维度的 token 位置。你应利用 token 位置，沿序列维度对（可能预先计算好的）cos 和 sin 张量做切片。

要测试你的实现，完成 `[adapters.run_rope]`，并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 Scaled Dot-Product Attention（缩放点积注意力）

现在，我们按 A. Vaswani et al. [8]（第 3.2.1 节）的描述实现 scaled dot-product attention。作为预备步骤，Attention 操作的定义会用到 softmax——一个把未归一化的分数向量转成归一化分布的操作：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n}\exp(v_j)} \tag{10}$$

注意，当值很大时，$\exp(v_i)$ 可能会变成 `inf`（于是 $\frac{inf}{inf}=\text{NaN}$）。我们可以利用 softmax 对给所有输入加任意常数 $c$ 不变的这一性质来避免这一点。我们可以利用这个性质来保证数值稳定性——通常，我们会从 $v$ 的所有元素中减去 $v$ 的最大项，使新的最大项变成 0。现在你将实现 softmax，并使用这个技巧来保证数值稳定性。

**Problem (softmax)：实现 softmax（1 分）**

> **交付物**：写一个函数，对张量应用 softmax 操作。你的函数应接受两个参数：一个张量和一个维度 $i$，并对输入张量的第 $i$ 维应用 softmax。输出张量应与输入张量同形状，但其第 $i$ 维现在是一个归一化的概率分布。使用"从第 $i$ 维所有元素中减去第 $i$ 维最大值"的技巧来避免数值稳定性问题。

要测试你的实现，完成 `[adapters.run_softmax]`，并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以把 Attention 操作在数学上定义如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{11}$$

其中 $Q\in\mathbb{R}^{n\times d_k}$，$K\in\mathbb{R}^{m\times d_k}$，$V\in\mathbb{R}^{m\times d_v}$。这里，$Q$、$K$ 和 $V$ 都是这个操作的输入——注意，它们不是可学习参数。

**掩码（masking）**：有时为 attention 操作的输出加一个掩码会很方便。掩码的形状应为 $M\in\{\text{True},\text{False}\}^{n\times m}$，这个布尔矩阵的每一行 $i$ 指示查询 $i$ 应当关注哪些 key。规范地（也略有些令人困惑地）讲，位置 $(i,j)$ 的值为 True 表示查询 $i$ **确实**关注 key $j$，值为 False 表示该查询不关注该 key。换句话说，"信息流动"发生在值为 True 的 $(i,j)$ 对上。例如，考虑一个 $1\times 3$ 的掩码矩阵，元素为 `[[True, True, False]]`，则唯一的那个查询向量只关注前两个 key。从计算上讲，使用掩码比在子序列上计算 attention 高效得多，我们可以这样做：取 pre-softmax 的值（$QK^\top/\sqrt{d_k}$），给掩码矩阵中所有 False 的项加上 $-\infty$。

**Problem (scaled_dot_product_attention)：实现 scaled dot-product attention（5 分）**

> **交付物**：实现 scaled dot-product attention 函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的 key 和 query，以及形状为 `(batch_size, ..., seq_len, d_v)` 的 value，其中 `...` 代表任意数量的其他批式维度（如果有的话）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于批式维度的讨论见第 3.2 节。

你的实现还应支持一个可选的、用户提供的、形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 True 的位置的注意力概率之和应为 1，掩码值为 False 的位置的注意力概率应为 0。

要针对我们提供的测试来检验你的实现，你需要实现测试适配器 `[adapters.run_scaled_dot_product_attention]`。`uv run pytest -k test_scaled_dot_product_attention` 会在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 会在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将按 A. Vaswani et al. [8] 第 3.2.2 节的描述实现多头自注意力。回顾一下，从数学上讲，应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \tag{12}$$

其中

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \tag{13}$$

$Q_i$、$K_i$、$V_i$ 分别是 $Q$、$K$、$V$ 的 embedding 维度中大小为 $d_k$ 或 $d_v$ 的第 $i\in\{1,\dots,h\}$ 个切片，而 Attention 是第 3.4.4 节定义的 scaled dot-product attention 操作。由此，我们可以构造多头自注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W_O\,\text{MultiHead}(W_Q x, W_K x, W_V x) \tag{14}$$

这里，可学习参数是 $W_Q\in\mathbb{R}^{h d_k\times d_{model}}$、$W_K\in\mathbb{R}^{h d_k\times d_{model}}$、$W_V\in\mathbb{R}^{h d_v\times d_{model}}$ 和 $W_O\in\mathbb{R}^{d_{model}\times h d_v}$。由于 $Q$、$K$、$V$ 在多头注意力操作中被切片，我们可以把 $W_Q$、$W_K$、$W_V$ 看作沿输出维度为每个头分开。当你把这些都实现好之后，你应当能用总共三次矩阵乘法完成 key、value 和 query 的投影。⁵

**因果掩码（causal masking）**

你的实现应阻止模型关注序列中未来的 token。换句话说，如果给模型一个 token 序列 $t_1,\dots,t_n$，而我们要为前缀 $t_1,\dots,t_i$（其中 $i<n$）计算下一个词的预测，模型就不应能访问（关注）位置 $t_{i+1},\dots,t_n$ 处的 token 表示，因为它在推理时生成文本时是访问不到这些 token 的（而且这些未来 token 会泄露真实下一个词的身份信息，让语言建模预训练目标变得平凡）。对于输入 token 序列 $t_1,\dots,t_n$，我们可以朴素地通过运行 $n$ 次多头自注意力（对应序列中的 $n$ 个唯一前缀）来阻止对未来 token 的访问。取而代之，我们使用因果注意力掩码，它允许 token $i$ 关注序列中所有 $j\le i$ 的位置。你可以用 `torch.triu` 或广播的索引比较来构造这个掩码，并且应该利用你的第 3.4.4 节 scaled dot-product attention 实现已经支持注意力掩码这一点。

**应用 RoPE**

RoPE 应应用到 query 和 key 向量上，但不应应用到 value 向量上。此外，头维度应作为批维度处理，因为在多头注意力中，每个头是独立地应用注意力的。这意味着，对每个头，都应对 query 和 key 向量应用完全相同的 RoPE 旋转。

**Problem (multihead_self_attention)：实现因果多头自注意力（5 分）**

> **交付物**：把因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应至少接受以下参数：

- `d_model: int` —— Transformer block 输入的维度。
- `num_heads: int` —— 多头自注意力中使用的头的数量。遵循 A. Vaswani et al. [8]，设 $d_k=d_v=\frac{d_{model}}{h}$。

要针对我们提供的测试来检验你的实现，实现测试适配器 `[adapters.run_multihead_self_attention]`。然后运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.5 完整的 Transformer LM

让我们先组装 Transformer block（回看图 2 会有帮助）。一个 Transformer block 包含两个"子层"，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层里，我们先做 RMSNorm，再执行主要操作（MHA/FF），最后加上残差连接。

具体地说，Transformer block 的前半部分（第一个"子层"）应实现下面这组更新，从输入 $x$ 产生输出 $y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)) \tag{15}$$

**Problem (transformer_block)：实现 Transformer block（3 分）**

按第 3.4 节的描述、图 2 的示意，实现 pre-norm Transformer block。你的 Transformer block 应至少接受以下参数：

- `d_model: int` —— Transformer block 输入的维度。
- `num_heads: int` —— 多头自注意力中使用的头的数量。
- `d_ff: int` —— 逐位置前馈内层的维度。

要测试你的实现，实现适配器 `[adapters.run_transformer_block]`。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。

> **交付物**：通过所提供测试的 Transformer block 代码。

现在，我们按照图 1 的高层示意把这些 block 组装起来。遵循第 3.1 节对 embedding 的描述，把输入送进 `num_layers` 个 Transformer block，再送入最后的 layer norm 和 LM head，得到词表上的未归一化分布（logits）。

**Problem (transformer_lm)：实现 Transformer LM（3 分）**

是时候把所有东西拼起来了！按第 3.1 节的描述、图 1 的示意实现 Transformer 语言模型。至少，你的实现应接受上述所有 Transformer block 的构造参数，以及这些额外参数：

- `vocab_size: int` —— 词表大小，用于确定 token embedding 矩阵的维度。
- `context_length: int` —— 最大上下文长度，用于确定 RoPE 的 sin/cos buffer 的维度。
- `num_layers: int` —— 使用的 Transformer block 数量。

要针对我们提供的测试来检验你的实现，你需要先实现测试适配器 `[adapters.run_transformer_lm]`。然后运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。

> **交付物**：通过上述测试的 Transformer LM 模块。

**资源核算（Resource accounting）**

能够理解 Transformer 的各个部分如何消耗计算和内存，是很有用的。我们来走一遍做一些基本 "FLOPs 核算" 的步骤。Transformer 中绝大多数的 FLOPs 都是矩阵乘法，所以我们的核心思路很简单：

1. 列出 Transformer 一次前向传播中的所有矩阵乘法。
2. 把每个矩阵乘法换算成所需的 FLOPs。

对于第二步，下面这个事实会很有用：

> **规则**：给定 $A\in\mathbb{R}^{m\times n}$ 和 $B\in\mathbb{R}^{n\times p}$，矩阵-矩阵乘积 $AB$ 需要 $2mnp$ 次 FLOPs。

要看出这一点，注意 $(AB)[i,j]=A[i,:]\cdot B[:,j]$，这个点积需要 $n$ 次加法和 $n$ 次乘法（$2n$ 次 FLOPs）。然后，由于矩阵-矩阵乘积 $AB$ 有 $m\times p$ 个元素，总 FLOPs 就是 $(2n)(mp)=2mnp$。现在，在做下一个问题之前，先把你 Transformer block 和 Transformer LM 的每个组件过一遍，列出所有矩阵乘法及其相应的 FLOPs 开销，会很有帮助。

**Problem (transformer_accounting)：Transformer LM 资源核算（5 分）**

(a) 考虑一个使用我们作业架构的 GPT-2 XL 规模模型，配置如下：

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288（最接近 $\frac{8}{3}\times 1{,}600$ 的 64 的倍数）

假设我们用这个配置构造模型。这个模型有多少可训练参数？假设每个参数用单精度浮点表示，仅加载这个模型就需要多少内存？

> **交付物**：一到两句话作答。

(b) 找出完成我们 GPT-2 XL 形状模型一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列有 `context_length` 个 token。

> **交付物**：一个矩阵乘法列表（带描述），以及所需的总 FLOPs。

(c) 根据上面的分析，模型的哪些部分需要最多的 FLOPs？

> **交付物**：一到两句话作答。

(d) 用 GPT-2 small（12 层、768 `d_model`、12 头）、GPT-2 medium（24 层、1024 `d_model`、16 头）和 GPT-2 large（36 层、1280 `d_model`、20 头）重复你的分析。随着模型规模增大，Transformer LM 的哪些部分在总 FLOPs 中占比相对变大或变小？

> **交付物**：对每个模型，给出模型组件的分解及其相应 FLOPs（占一次前向传播总 FLOPs 的比例）。此外，用一到两句话描述改变模型规模如何改变每个组件的 FLOPs 占比。

(e) 取 GPT-2 XL，把上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型各组件 FLOPs 的相对贡献如何变化？

> **交付物**：一到两句话作答。

## 4 训练一个 Transformer LM

现在我们已经有了预处理数据的步骤（通过分词器）和模型（Transformer）。剩下要做的，就是构建训练所需的所有代码。这包括以下内容：

- **损失**：我们需要定义损失函数（交叉熵）。
- **优化器**：我们需要定义最小化这个损失的优化器（AdamW）。
- **训练循环**：我们需要所有支撑性基础设施，用于加载数据、保存 checkpoint、管理训练。

### 4.1 交叉熵损失

回顾一下，Transformer 语言模型为每个长度为 $m+1$ 的序列 $x$、以及 $i=1,\dots,m$，定义了一个分布 $p_\theta(x_{i+1}\mid x_{1:i})$。给定一个由长度为 $m+1$ 的序列构成的训练集 $D$，我们定义标准的交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D|\, m}\sum_{x\in D}\sum_{i=1}^{m} -\log p_\theta(x_{i+1}\mid x_{1:i}) \tag{16}$$

（注意，Transformer 的一次前向传播就会为所有 $i=1,\dots,m$ 产生 $p_\theta(x_{i+1}\mid x_{1:i})$。）

特别地，Transformer 为每个位置 $i$ 计算 logits $o_i\in\mathbb{R}^{vocab\_size}$，从而得到：⁶

$$p(x_{i+1}\mid x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size}\exp(o_i[a])} \tag{17}$$

交叉熵损失通常是相对于 logits 向量 $o_i\in\mathbb{R}^{vocab\_size}$ 和目标 $x_{i+1}$ 来定义的。⁷

实现交叉熵损失时，和 softmax 的情形一样，需要注意数值问题。

**Problem (cross_entropy)：实现交叉熵（1 分）**

> **交付物**：写一个计算交叉熵损失的函数，它接收预测的 logits（$o_i$）和目标（$x_{i+1}$），计算交叉熵 $\ell_i = -\log\text{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下事项：

- 减去最大元素以保证数值稳定性。
- 尽可能把 log 和 exp 消掉。
- 处理任意额外的批维度，并返回跨批的平均值。和第 3.2 节一样，我们假设批式维度总是排在最前面，位于词表大小维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

**Perplexity（困惑度）**

训练用交叉熵就够了，但在评估模型时，我们还希望报告 perplexity。对于一个长度为 $m$ 的序列，设其交叉熵损失为 $\ell_1,\dots,\ell_m$：

$$\text{perplexity} = \exp\left(\frac{1}{m}\sum_{i=1}^{m}\ell_i\right) \tag{18}$$

### 4.2 SGD 优化器

有了损失函数，我们开始探索优化器。最简单的基于梯度的优化器是 Stochastic Gradient Descent（SGD，随机梯度下降）。我们从随机初始化的参数 $\theta_0$ 开始，然后对每一步 $t=0,\dots,T-1$，执行如下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t) \tag{19}$$

其中 $B_t$ 是从数据集 $D$ 中采样的一个随机数据批，学习率 $\alpha_t$ 和批大小 $|B_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

要实现我们的优化器，我们会继承 PyTorch 的 `torch.optim.Optimizer` 类。一个 `Optimizer` 子类必须实现两个方法：

- `def __init__(self, params, ...)` 应初始化你的优化器。这里，`params` 是要优化的一组参数（或者是参数组，如果用户想对模型的不同部分使用不同的超参数，比如不同的学习率的话）。务必把 `params` 传给基类的 `__init__` 方法，基类会存储这些参数供 `step` 使用。你可以根据优化器接受额外的参数（学习率就是常见的一个），并以字典的形式把它们传给基类构造函数，其中键是你为这些参数选择的名字（字符串）。
- `def step(self)` 应执行一次参数更新。在训练循环中，这会在反向传播之后调用，因此你可以访问上一批的梯度。这个方法应遍历每个参数张量 `p` 并就地修改它们，即设置 `p.data`（它保存与该参数关联的张量），依据的是梯度 `p.grad`（如果存在的话）——它是损失关于该参数的梯度张量。

PyTorch 优化器 API 有一些微妙之处，所以用一个例子来解释会更简单。为了让例子更丰富，我们实现一个 SGD 的轻微变体：学习率在训练过程中衰减，从初始学习率 $\alpha$ 开始，随时间走越来越小的步：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}}\nabla L(\theta_t; B_t) \tag{20}$$

让我们看看这个 SGD 变体如何实现为一个 PyTorch Optimizer：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]           # 获取与 p 关联的状态。
                t = state.get("t", 0)           # 从状态中获取迭代次数，或默认为 0。
                grad = p.grad.data              # 获取损失关于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量。
                state["t"] = t + 1              # 递增迭代次数。

        return loss
```

在 `__init__` 中，我们把参数以及默认超参数传给基类构造函数（参数可能以组的形式传入，每组有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象的集合，基类构造函数会创建一个组，并赋予它默认超参数。然后，在 `step` 中，我们遍历每个参数组，再遍历该组中的每个参数，应用公式 20。这里，我们把迭代次数作为与每个参数关联的状态保存：先读取这个值，在梯度更新中使用它，然后更新它。API 规定，用户可能传入一个可调用的 `closure`，在优化器步进之前重新计算损失。对于我们使用的优化器来说并不需要这个，但我们加上它以符合 API。

要看到它工作，我们可以用下面这个最小的训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()                # 重置所有可学习参数的梯度。
    loss = (weights**2).mean()     # 计算一个标量损失值。
    print(loss.cpu().item())
    loss.backward()                # 运行反向传播，计算梯度。
    opt.step()                     # 运行优化器步进。
```

这是训练循环的典型结构：在每次迭代中，我们计算损失并运行一步优化器。训练语言模型时，我们的可学习参数来自模型（在 PyTorch 中，`m.parameters()` 会给我们这个集合）。损失是在一批采样数据上计算的，但训练循环的基本结构是相同的。

**Problem (learning_rate_tuning)：调学习率（1 分）**

如我们将看到的，对训练影响最大的超参数之一是学习率。让我们在这个玩具示例里实际看看。用另外三个学习率值运行上面的 SGD 示例：`1e1`、`1e2` 和 `1e3`，只跑 10 次训练迭代。这些学习率下损失分别发生了什么？它是衰减得更快、更慢，还是发散（即在训练过程中增大）？

> **交付物**：一到两句话作答，描述你观察到的行为。

### 4.3 AdamW

现代语言模型通常用更复杂的优化器训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生品。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛使用。AdamW 提出对 Adam 的一种修改，通过加入 weight decay（每次迭代把参数往 0 拉）来改善正则化，且其方式与梯度更新解耦。我们将按 I. Loshchilov et al. [23] 的算法 2 来实现 AdamW。

AdamW 是有状态的：对每个参数，它跟踪其一阶矩和二阶矩的运行估计。因此，AdamW 用额外的内存换取了更好的稳定性和收敛性。除了学习率 $\alpha$ 之外，AdamW 还有一对超参数 $(\beta_1,\beta_2)$，控制矩估计的更新，以及一个 weight decay 率 $\lambda$。典型应用把 $(\beta_1,\beta_2)$ 设为 $(0.9,0.999)$，但 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 等大语言模型通常用 $(0.9,0.95)$ 训练。算法可以写成如下形式，其中 $\varepsilon$ 是一个小值（例如 $10^{-8}$），用于在 $v$ 出现极小值时改善数值稳定性：

**算法 1：AdamW 优化器**

```
 1  init(θ)                                     ▷ 初始化可学习参数
 2  m ← 0                                       ▷ 一阶矩向量的初始值；与 θ 同形状
 3  v ← 0                                       ▷ 二阶矩向量的初始值；与 θ 同形状
 4  for t = 1, …, T do
 5      Sample batch of data B_t
 6      g ← ∇_θ ℓ(θ; B_t)                       ▷ 计算损失的梯度
 7      α_t ← α √(1-β_2^t)/(1-β_1^t)            ▷ 计算迭代 t 的调整后 α
 8      θ ← θ − αλθ                             ▷ 应用 weight decay
 9      m ← β_1 m + (1 − β_1) g                 ▷ 更新一阶矩估计
10      v ← β_2 v + (1 − β_2) g²                ▷ 更新二阶矩估计
11      θ ← θ − α_t m/(√v + ε)                  ▷ 应用矩调整后的权重更新
12  end for
```

**算法 1：AdamW 优化器伪代码。**

注意，$t$ 从 1 开始。现在你将实现这个优化器。

**Problem (adamw)：实现 AdamW（2 分）**

> **交付物**：把 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接收学习率 $\alpha$，以及 $\beta$、$\varepsilon$ 和 $\lambda$ 超参数。为了帮你保存状态，基类 `Optimizer` 给了你一个字典 `self.state`，它把 `nn.Parameter` 对象映射到一个字典，后者存储你为该参数需要的任何信息（对 AdamW 来说，就是矩估计）。实现 `[adapters.get_adamw_cls]`，并确保它通过 `uv run pytest -k test_adamw`。

**Problem (adamw_accounting)：AdamW 训练的资源核算（2 分）**

让我们计算运行 AdamW 需要多少内存和计算。假设每个张量都用 float32。

(a) 运行 AdamW 需要多少峰值内存？请按参数、激活值、梯度和优化器状态的内存占用分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）来表达你的答案。假设 $d_{ff}=\frac{8}{3}\times d_{model}$。为简单起见，计算激活值内存占用时，只考虑以下组件：

- Transformer block
  - RMSNorm(s)
  - 多头自注意力子层：QKV 投影、$QK^\top$ 矩阵乘法、softmax、value 的加权和、输出投影。
  - 逐位置前馈（SwiGLU）：$W_1$、$W_2$、门控分支上的 SiLU、逐元素乘积、$W_3$
- 最终的 RMSNorm
- 输出 embedding
- logits 上的交叉熵

> **交付物**：参数、激活值、梯度和优化器状态各自的代数表达式，以及总和。

(b) 把你的答案代入 GPT-2 XL 形状的模型，得到一个只依赖 `batch_size` 的表达式。在 80GB 内存内能使用的最大 batch size 是多少？

> **交付物**：一个形如 $a\cdot batch\_size + b$ 的表达式（其中 $a$、$b$ 是数值），以及一个表示最大 batch size 的数字。

(c) 运行一步 AdamW 需要多少 FLOPs？

> **交付物**：一个代数表达式，附带简要论证。

(d) Model FLOPs utilization（MFU，模型 FLOPs 利用率）定义为观测吞吐量（每秒 token 数）相对于硬件理论峰值 FLOP 吞吐量的比率 [A. Chowdhery et al., 2022]。一块 NVIDIA H100 GPU 对 "float32"（实际上是 TensorFloat-32，现实中是 "bfloat19"）操作的理论峰值是 495 teraFLOP/s。假设你能达到 50% 的 MFU，在一块 H100 上训练一个 GPT-2 XL 400K 步、batch size 为 1024，需要多长时间？遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设反向传播的 FLOPs 是前向传播的两倍。

> **交付物**：训练所需的小时数，附带简要论证。

### 4.4 学习率调度

能让损失下降最快的学习率值，在训练过程中往往是变化的。在训练 Transformer 时，通常会使用一个学习率调度表：开始时用较大的学习率，在初期做出更快的更新，然后随着模型训练，逐渐把它衰减到一个较小的值。⁸ 在本次作业中，我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的余弦退火调度。

调度器本质上就是一个函数，接收当前步 $t$ 和其他相关参数（如初始和最终学习率），返回第 $t$ 步梯度更新所用的学习率。最简单的调度是常数函数，对任意 $t$ 都返回同一个学习率。

余弦退火学习率调度接收：(i) 当前迭代 $t$，(ii) 最大学习率 $\alpha_{max}$，(iii) 最小（最终）学习率 $\alpha_{min}$，(iv) warm-up 迭代次数 $T_w$，以及 (v) 余弦退火的最终迭代 $T_c$。第 $t$ 次迭代的学习率定义如下：

- （Warm-up）若 $t < T_w$，则 $\alpha_t = \frac{t}{T_w}\alpha_{max}$。
- （余弦退火）若 $T_w \le t \le T_c$，则 $\alpha_t = \alpha_{min} + \frac{1}{2}\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{max}-\alpha_{min})$。
- （退火之后）若 $t > T_c$，则 $\alpha_t = \alpha_{min}$。

**Problem (learning_rate_schedule)：实现带 warmup 的余弦学习率调度（1 分）**

写一个函数，接收 $t$、$\alpha_{max}$、$\alpha_{min}$、$T_w$ 和 $T_c$，按上面定义的调度器返回学习率 $\alpha_t$。然后实现 `[adapters.get_lr_cosine_schedule]`，并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪（gradient clipping）

训练过程中，我们有时会遇到产生巨大梯度的训练样本，这可能破坏训练的稳定性。为了缓解这一点，实践中常用的一种技术是梯度裁剪。其思路是：在每次反向传播之后、执行优化器步进之前，对梯度的范数施加一个限制。给定（所有参数的）梯度 $g$，我们计算它的 $\ell_2$ 范数 $\|g\|_2$。如果这个范数小于最大值 $M$，就让 $g$ 保持不变；否则，我们把 $g$ 缩小一个因子 $\frac{M}{\|g\|_2+\varepsilon}$（其中加一个小 $\varepsilon$，如 $10^{-6}$，用于数值稳定性）。注意，得到的范数会略小于 $M$。

**Problem (gradient_clipping)：实现梯度裁剪（1 分）**

写一个实现梯度裁剪的函数。你的函数应接收一个参数列表和一个最大 $\ell_2$ 范数，并就地修改每个参数的梯度。使用 $\varepsilon=10^{-6}$（PyTorch 默认值）。然后，实现适配器 `[adapters.run_gradient_clipping]`，并确保它通过 `uv run pytest -k test_gradient_clipping`。

## 5 训练循环

现在，我们终于要把迄今为止构建的主要组件拼起来了：分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据（例如你在 `tokenizer_experiments` 中准备的数据）是一个 token 序列 $x=(x_1,\dots,x_n)$。尽管源数据可能由独立的文档组成（例如不同的网页或源代码文件），常见的做法是把它们全部拼接成单个 token 序列，在它们之间加上分隔符（如 `<|endoftext|>` token）。数据加载器把它变成一批批数据流，其中每个 batch 由 $B$ 个长度为 $m$ 的序列组成，并与对应的下一个 token 配对，长度也是 $m$。例如，对于 $B=1$、$m=3$，$([x_2,x_3,x_4],[x_3,x_4,x_5])$ 就是一个可能的 batch。

以这种方式加载数据简化了训练，原因有几个。首先，任意 $1\le i\le n-m$ 都给出一个有效的训练序列，所以采样训练序列很trivial（简单）。由于所有训练序列长度相同，就不需要对输入序列做 padding，这提升了硬件利用率（也通过增大 batch size $B$ 实现）。最后，我们也不需要加载完整数据集来采样训练数据，从而便于处理那些可能放不进内存的大数据集。

**Problem (data_loading)：实现数据加载（2 分）**

> **交付物**：写一个函数，接收一个 numpy 数组 $x$（token ID 的整数数组）、一个 `batch_size`、一个 `context_length` 和一个 PyTorch 设备字符串（如 `'cpu'` 或 `'cuda:0'`），返回一对张量：采样的输入序列和对应的下一个 token 目标。两个张量都应为形状 `(batch_size, context_length)`，包含 token ID，且都应放到请求的设备上。要针对我们提供的测试来检验你的实现，你需要先实现测试适配器 `[adapters.run_get_batch]`。然后运行 `uv run pytest -k test_get_batch` 来测试你的实现。

> **低资源提示：在 CPU 或 Apple Silicon 上做数据加载**
>
> 如果你打算在 CPU 或 Apple Silicon 上训练 LM，你需要把数据移到正确的设备上（同理，之后模型也应使用相同的设备）。
>
> 如果你在 CPU 上，可以用 `'cpu'` 设备字符串；在 Apple Silicon（M\* 芯片）上，可以用 `'mps'` 设备字符串。关于 MPS 的更多信息，请查看这些资源：
>
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

如果数据集太大放不进内存怎么办？我们可以用一个名为 `mmap` 的 Unix 系统调用，它把磁盘上的文件映射到虚拟内存，并在访问到那个内存位置时惰性地加载文件内容。这样，你就可以"假装"把整个数据集都放在内存里了。Numpy 通过 `np.memmap` 实现这一点（或者，如果你当初是用 `np.save` 保存数组的，也可以用 `np.load` 的 `mmap_mode='r'` 标志），它会返回一个类似 numpy 数组的对象，在你访问时按需加载条目。在训练中从数据集（即 numpy 数组）采样时，务必以内存映射模式加载数据集（根据你保存数组的方式，用 `np.memmap` 或 `np.load` 的 `mmap_mode='r'` 标志）。务必指定一个与你加载的数组匹配的 dtype。显式验证一下内存映射的数据看起来是否正确（例如，不包含超出预期词表大小的值），可能会很有帮助。

### 5.2 Checkpoint（检查点）

除了加载数据，我们还需要在训练过程中保存模型。运行作业时，我们常常希望能在中途停下来的训练上恢复（例如，由于作业超时、机器故障等）。即便一切顺利，我们也可能希望之后能访问中间模型（例如事后研究训练动态、从不同训练阶段的模型采样等）。

一个 checkpoint 应包含恢复训练所需的全部状态。我们当然至少要能恢复模型权重。如果使用有状态的优化器（如 AdamW），我们还需要保存优化器的状态（例如 AdamW 情况下的矩估计）。最后，为了恢复学习率调度，我们需要知道停在了哪个迭代次数。PyTorch 让保存所有这些变得简单：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回包含所有可学习权重的字典；我们稍后可以用姊妹方法 `load_state_dict()` 恢复这些权重。任何 `torch.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以把一个对象（例如，一个包含张量作为某些值、但也包含整数等普通 Python 对象的字典）转储到文件（路径）或类文件对象，之后可以用 `torch.load(src)` 加载回内存。

**Problem (checkpointing)：实现模型 checkpoint（1 分）**

实现下面两个函数来加载和保存 checkpoint：

```
def save_checkpoint(model, optimizer, iteration, out)
```

应把模型、优化器和迭代次数的所有状态转储到类文件对象 `out` 中。你可以用模型和优化器的 `state_dict` 方法获取它们相关的状态，并用 `torch.save(obj, out)` 把 `obj` 转储到 `out`（PyTorch 在这里既支持路径也支持类文件对象）。一个典型的选择是让 `obj` 是一个字典，但只要你能在之后加载你的 checkpoint，用任何格式都可以。该函数期望以下参数：

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```
def load_checkpoint(src, model, optimizer)
```

应从 `src`（路径或类文件对象）加载一个 checkpoint，然后从该 checkpoint 恢复模型和优化器的状态。你的函数应返回保存到 checkpoint 中的迭代次数。你可以用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，并用模型和优化器的 `load_state_dict` 方法把它们恢复到之前的状态。

该函数期望以下参数：

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在，终于到了把你实现的所有组件拼进主训练脚本的时候了。让你能够轻松地以不同超参数启动训练（例如把它们作为命令行参数接收），是值得的，因为之后你会为了研究不同选择如何影响训练而做很多次。

**Problem (training_together)：把它们拼起来（4 分）**

> **交付物**：写一个脚本，运行训练循环，在用户提供的输入上训练你的模型。特别地，我们建议你的训练脚本至少支持以下能力：

- 能够配置和控制各种模型和优化器超参数。
- 用 `np.memmap` 高效地加载大型训练和验证数据集。
- 把 checkpoint 序列化到用户提供的路径。
- 周期性记录训练和验证性能（例如输出到控制台和/或 Weights and Biases 之类的外部服务）。⁹

## 6 生成文本

现在我们已经能训练模型了，最后一块拼图是能够从模型生成文本。回顾一下，语言模型接收一个（可能分批的）长度为 `sequence_length` 的整数序列，产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素都是一个预测该位置之后下一个 token 的概率分布。现在我们要写几个函数，把它转成对新序列的采样方案。

**Softmax**

按照标准惯例，语言模型的输出是最终线性层（"logits"）的输出，所以我们必须通过 softmax 操作把它转成归一化概率，这我们在前面的公式 10 里已经见过。

**解码（Decoding）**

要从模型生成（解码）文本，我们给模型一个前缀 token 序列（"prompt"），让它产生一个预测序列中下一个 token 的词表概率分布。然后，我们从这个词表项分布中采样，来确定下一个输出 token。

具体地，解码过程的一步应接收序列 $x_{1\dots t}$，并通过以下公式返回 token $x_{t+1}$：

$$P(x_{t+1}=i \mid x_{1\dots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)} \tag{21}$$

$$v = \text{TransformerLM}(x_{1\dots t})_t \in \mathbb{R}^{vocab\_size} \tag{22}$$

其中 TransformerLM 是我们的模型，它接收一个长度为 `sequence_length` 的序列，产生一个大小为 `(sequence_length, vocab_size)` 的矩阵；我们取这个矩阵的最后一个元素，因为我们要求的是第 $t$ 个位置的下一个 token 预测。这样，通过反复从这些一步条件分布中采样（把之前生成的输出 token 追加到下一个解码时间步的输入中），直到生成序列结束 token `<|endoftext|>`（或用户指定的最大生成 token 数），我们就得到了一个基本的解码器。

**解码技巧**

我们将用小型模型做实验，而小模型有时会生成质量很低的文本。两个简单的解码技巧可以帮助解决这些问题。首先，在 temperature scaling（温度缩放）中，我们用一个温度参数 $\tau$ 修改 softmax，新的 softmax 是：

$$\text{softmax}(v,\tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{vocab\_size}\exp(v_j/\tau)} \tag{23}$$

注意，当 $\tau\to 0$ 时，$v$ 的最大元素会主导，softmax 的输出会变成集中在这个最大元素上的 one-hot 向量。

其次，另一个技巧是 nucleus（核）或 top-p 采样，我们通过截断低概率 token 来修改采样分布。设 $q$ 是从（经过温度缩放的）softmax 得到的、大小为 `vocab_size` 的概率分布。带超参数 $p$ 的 nucleus 采样按以下公式产生下一个 token：

$$P(x_{t+1}=i \mid q) = \begin{cases} \frac{q_i}{\sum_{j\in V^{(p)}} q_j} & \text{if } i\in V^{(p)} \\ 0 & \text{otherwise} \end{cases} \tag{24}$$

其中 $V^{(p)}$ 是满足 $\sum_{j\in V^{(p)}} q_j \ge p$ 的最小索引集合。你可以这样轻松计算这个量：先把概率分布 $q$ 按大小排序，然后选取最大的词表元素，直到达到目标水平 $p$。

**Problem (decoding)：解码（3 分）**

> **交付物**：实现一个从你的语言模型解码的函数。我们建议你支持以下功能：

- 为用户提供的 prompt 生成补全（即接收某个 $x_{1\dots t}$，采样补全，直到遇到 `<|endoftext|>` token）。
- 允许用户控制生成 token 的最大数量。
- 给定所需的温度值，在采样前对预测的下一个 token 分布应用 softmax 温度缩放。
- Top-$p$ 采样（A. Holtzman et al. [2020]，也称为 nucleus 采样），给定一个用户指定的阈值。

## 7 实验

现在是把所有东西拼起来、在预训练数据集上训练（小型）语言模型的时候了。

### 7.1 如何运行实验与交付物

理解 Transformer 各架构组件背后原理的最好方式，就是实际去修改它、运行它。亲自动手的经验无可替代。

为此，能够快速、一致地做实验，并记录你所做的，是很重要的。为了快速实验，我们会在一个小规模模型（总共约 17M 参数）和简单数据集（TinyStories）上跑很多实验。为了一致地做事，你会系统性地消融组件、改变超参数；为了记录，我们会要求你提交一份实验日志，以及与每个实验相关的学习曲线。为了能够提交损失曲线，务必周期性评估验证损失，并记录步数和 wall-clock（墙钟）时间。你可能会发现 Weights and Biases 之类的日志基础设施很有帮助。

**Problem (experiment_log)：实验日志（3 分）**

为你的训练和评估代码创建实验跟踪基础设施，使你能够跟踪实验和损失曲线（相对于梯度步数和 wall-clock 时间）。

> **交付物**：用于实验的日志基础设施代码，以及本节下面作业问题的一份实验日志（记录你所尝试的一切的文档）。

### 7.2 TinyStories

我们将从一个非常简单的数据集（TinyStories；R. Eldan et al. [1]）开始，模型会训练得很快，而且我们能观察到一些有趣的行为。获取该数据集的说明在第 1 节。下面是这个数据集样子的一个例子。

**Example (tinystories_example)：来自 TinyStories 的一个例子**

> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

#### 7.2.1 超参数调优

我们会告诉你一些非常基础的起始超参数，并让你为其他超参数找到好用的设置。

- **词表大小 10000。** 典型的词表大小在几万到几十万之间。你应该改变它，看看词表和模型行为如何变化。
- **上下文长度 256。** 像 TinyStories 这样的简单数据集可能不需要很长的序列长度，但对于后面的 OpenWebText 数据，你可能想改变它。试着改变它，看看对每次迭代运行时间和最终 perplexity 的影响。
- **d_model 512。** 这比许多小型 Transformer 论文中使用的 768 维略小，但这会让速度更快。
- **d_ff 1344。** 这大约是 $\frac{8}{3}d_{model}$，同时是 64 的倍数，这对 GPU 性能有利。
- **RoPE theta 参数 Θ 10000。**
- **层数和头数：4 层、16 头。** 合起来将给出约 17M 的非 embedding 参数，这是一个相当小的 Transformer。**处理的总 token 数 327,680,000**（你的 batch size × 总步数 × 上下文长度应大致等于这个值）。

你应该通过试错，为以下其他超参数找到好的默认值：学习率、学习率 warmup、其他 AdamW 超参数（$\beta_1$、$\beta_2$、$\varepsilon$）以及 weight decay。你可以在 D. P. Kingma et al. [22] 中找到这些超参数的一些典型选择。

#### 7.2.2 拼起来

现在，你可以通过获取一个训练好的 BPE 分词器、分词训练数据集，然后在你写的训练循环里运行，来把所有东西拼起来了。重要提示：如果你的实现正确且高效，上述超参数在 1 块 B200 GPU 上应能带来大约 20–30 分钟的运行时间。如果你的运行时间远长于此，请检查并确保你的数据加载、checkpoint 或验证损失代码没有成为瓶颈，而且你的实现是正确分批的。

#### 7.2.3 调试模型架构的技巧

我们强烈建议你熟悉 IDE 内置的调试器（如 VSCode/Zed），这比用 print 语句调试更省时间。如果你用文本编辑器，可以用 `ipdb` 之类的东西。调试模型架构时还有几个好习惯：

- 开发任何神经网络架构时，一个常见的起步做法是过拟合到单个 minibatch。如果你的实现正确，你应该能很快把训练损失压到接近零。
- 在各种模型组件里设置调试断点，检查中间张量的形状，确保它们符合预期。
- 监控激活值、模型权重和梯度的范数，确保它们没有爆炸或消失。

**Problem (learning_rate)：调学习率（2 B200 hrs）（3 分）**

学习率是最需要调优的超参数之一。以你训练好的基础模型为对象，回答以下问题：

(a) 对学习率做一次超参数搜索，报告最终损失（如果优化器发散，则注明发散）。

> **交付物**：与多个学习率相关的学习曲线。解释你的超参数搜索策略。
>
> **交付物**：一个在 TinyStories 上（每 token）验证损失不超过 1.45 的模型。

> **低资源提示：在 CPU 或 Apple Silicon 上训练几步**
>
> 如果你在 cpu 或 mps 上运行，你应该把处理的总 token 数减少到 40,000,000，这足以产生相当流畅的文本。你也可以把目标验证损失从 1.45 提高到 2.00。用调好学习率的参考代码，在 M4 Max 芯片和 36 GB 内存上，我们使用 batch size × 总步数 × 上下文长度 = 32 × 5000 × 256 = 40,960,000 token，在 cpu 上耗时 1 小时 22 分，在 mps 上 36 分钟。在第 5000 步，我们达到了 1.80 的验证损失。
>
> 一些额外的提示：
>
> - 当使用 $N$ 个训练步时，我们建议调整余弦学习率衰减调度，使其恰好在第 $N$ 步结束衰减（即达到最小学习率）。
> - 使用 mps 时，不要使用 TF32 kernel，即不要像在 cuda 设备上那样设置 `torch.set_float32_matmul_precision('high')`。我们尝试过用 mps（torch 版本 2.9.0）启用 TF32 kernel，发现后端有时会使用静默损坏的 kernel，导致训练不稳定。
> - 你可以通过用 `torch.compile` 对模型做 JIT 编译来加速训练。具体地：
>   - 在 cpu 上，用 `model = torch.compile(model)` 编译你的模型。
>   - 在 mps 上，你可以用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。截至 torch 版本 2.9.0，mps 上不支持用 Inductor 编译。

(b) 民间智慧说，最好的学习率"处在稳定性的边缘"。研究一下学习率发散的点与你最佳学习率之间的关系。

> **交付物**：递增学习率的学习曲线，其中至少包含一次发散运行，以及关于这与收敛速度之间关系的分析。

现在让我们改变 batch size，看看训练会发生什么。batch size 很重要——通过做更大的矩阵乘法，它们能让我们从 GPU 上获得更高效率，但"我们总是想要大 batch size"这个说法真的成立吗？让我们跑一些实验来找出答案。

**Problem (batch_size_experiment)：batch size 变化（1 B200 hr）（1 分）**

把你的 batch size 从 1 一路变到 GPU 内存极限。中间至少试几个 batch size，包括 64 和 128 这样的典型值。

> **交付物**：不同 batch size 运行的学习曲线。如有必要，学习率应重新优化。
>
> **交付物**：几句话，讨论你关于 batch size 及其对训练影响的发现。

有了解码器，我们现在就可以生成文本了！我们将从模型生成，看看它有多好。作为参考，你应该得到至少和下面例子一样好的输出。

**Example (ts_generate_example)：来自 TinyStories 语言模型的采样输出**

> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 如果你使用的是处理了 40M token 的低资源配置，你看到的生成结果应该仍然像英语，但不如上面的流畅。例如，我们一个在 40M token 上训练的 TinyStories 语言模型的采样输出如下：
>
> > Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together. "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!" As Sue watched the sky moved, what it was. She

这里是精确的问题陈述和我们的要求：

**Problem (generate)：生成文本（1 分）**

用你的解码器和训练好的 checkpoint，报告你的模型生成的文本。你可能需要调整解码器参数（temperature、top-p 等）来获得流畅的输出。

> **交付物**：至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），以及对这段输出流畅度的简短评论，以及至少两个影响这段输出好坏的因子。

### 7.3 消融与架构修改

理解 Transformer 的最好方式，就是实际修改它、看看它的行为。现在，我们来做几个简单的消融和修改。

**消融 1：layer normalization**

人们常说 layer normalization 对 Transformer 训练的稳定性很重要。但也许我们想冒险一点。让我们从每个 Transformer block 中移除 RMSNorm，看看会发生什么。

**Problem (layer_norm_ablation)：移除 RMSNorm 并训练（0.5 B200 hrs）（1 分）**

从你的 Transformer 中移除所有 RMSNorm 并训练。在之前的最优学习率下会发生什么？你能通过使用更低的学习率来获得稳定性吗？

> **交付物**：移除 RMSNorm 后训练的学习曲线，以及最佳学习率下的学习曲线。
>
> **交付物**：几句关于 RMSNorm 影响的评论。

现在，让我们研究另一种乍一看似乎很随意的 layer normalization 选择。Pre-norm Transformer block 定义为

$$z = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)) \tag{25}$$
$$y = z + \text{FFN}(\text{RMSNorm}(z)) \tag{26}$$

这是对原始 Transformer 架构为数不多的"共识性"修改之一，原架构使用的是 post-norm 方式：

$$z = \text{RMSNorm}(x + \text{MultiHeadSelfAttention}(x)) \tag{27}$$
$$y = \text{RMSNorm}(z + \text{FFN}(z)) \tag{28}$$

让我们回到 post-norm 方式，看看会发生什么。

**Problem (pre_norm_ablation)：实现 post-norm 并训练（0.5 B200 hrs）（1 分）**

把你的 pre-norm Transformer 实现修改成 post-norm 的。用 post-norm 模型训练，看看会发生什么。

> **交付物**：post-norm Transformer 的学习曲线，与 pre-norm 的对比。

我们看到，layer normalization 对 Transformer 的行为有重大影响，而且即便是 layer normalization 的位置也很重要。

**消融 2：位置嵌入**

接下来，我们将研究位置嵌入对模型性能的影响。具体地，我们将比较基础模型（带 RoPE）和完全不包含位置嵌入（NoPE）。事实证明，decoder-only Transformer（即像我们实现的那样带因果掩码的），理论上可以在没有显式提供位置嵌入的情况下，推断出相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。现在我们将实证地测试 NoPE 与 RoPE 相比表现如何。

**Problem (no_pos_emb)：实现 NoPE（0.5 B200 hrs）（1 分）**

修改你带 RoPE 的 Transformer 实现，彻底移除位置嵌入信息，看看会发生什么。

> **交付物**：比较 RoPE 和 NoPE 性能的学习曲线。

**消融 3：SwiGLU vs. SiLU**

接下来，我们将遵循 N. Shazeer [20]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有 gated linear unit（GLU）的前馈网络，来测试前馈网络中门控的重要性：

$$\text{FFN}_{\text{SiLU}}(x) = W_2\,\text{SiLU}(W_1 x) \tag{29}$$

回顾一下，在我们的 SwiGLU 实现中，我们把前馈内层的维度设为约 $d_{ff}=\frac{8}{3}d_{model}$（同时确保 $d_{ff} \bmod 64 = 0$，以利用 GPU tensor core）。在这个消融基线中，你的 FFN_SiLU 实现应改为设 $d_{ff}=4\times d_{model}$，以大致匹配默认 SwiGLU 前馈网络的参数数量（它有三个而不是两个权重矩阵）。

**Problem (swiglu_ablation)：SwiGLU vs. SiLU（0.5 B200 hrs）（1 分）**

> **交付物**：比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，参数数量大致匹配。
>
> **交付物**：几句话讨论你的发现。

> **低资源提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
>
> 在作业的剩余部分，我们将转向一个更大规模、更嘈杂的网络数据集（OpenWebText），实验架构修改，并（可选地）向课程排行榜提交。在 OpenWebText 上把 LM 训练到流畅需要很长时间，所以我们建议 GPU 访问受限的在线学生继续在 TinyStories 上测试修改（用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

现在，我们将转向一个由网络爬取构建的、更标准的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一个小样本也作为单个文本文件提供：如何访问该文件见第 1 节。下面是 OpenWebText 的一个例子。注意，文本要真实、复杂、多样得多。你可能想浏览一下训练数据集，感受一下网络抓取语料的训练数据长什么样。

**Example (owt_example)：来自 OWT 的一个例子**

> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job. "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. [...]

> **注意**：这个实验你可能需要重新调优学习率或 batch size 等超参数。

**Problem (main_experiment)：在 OWT 上实验（2 B200 hrs）（2 分）**

用与 TinyStories 相同的模型架构和总训练迭代次数，在 OpenWebText 上训练你的语言模型。这个模型表现如何？

> **交付物**：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 的损失差异——我们应该如何解读这些损失？
>
> **交付物**：来自 OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？为什么即使模型和算力预算与 TinyStories 相同，输出质量却更差？

### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你差不多完成了！现在你将尝试改进 Transformer 架构，看看你的超参数和架构与班上其他同学相比如何。

**排行榜规则**

除了以下几点外没有其他限制：

- **运行时间**：你的提交在 B200 上最多运行 45 分钟。如果你使用 SLURM 或 Modal，可能想在提交脚本中强制执行这一点。
- **数据**：你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你可以随心所欲。

如果你想找一些实现的灵感，可以看看这些资源：

- 最先进的开源 LLM 家族，如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库（`github.com/KellerJordan/modded-nanogpt`），社区成员在这里发布了许多用于"速通"小规模语言模型预训练的有趣修改。例如，一个可追溯到原始 Transformer 论文的常见修改是，把输入和输出 embedding 的权重绑定起来（见 A. Vaswani et al. [8]（第 3.4 节）和 A. Chowdhery et al. [16]（第 2 节））。如果你尝试权重绑定，可能需要降低 embedding/LM head 初始化的标准差。在尝试完整的 45 分钟运行之前，你应该先在 OpenWebText 的一个小子集或 TinyStories 上测试这些。

作为一个提醒，我们确实注意到，你在排行榜上发现奏效的一些修改，可能无法推广到更大规模的预训练。我们将在课程的 scaling laws 单元中进一步探讨这个想法。

**Problem (leaderboard)：排行榜（10 B200 hrs）（6 分）**

你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200-hours 内最小化你的语言模型的验证损失。

> **交付物**：记录到的最终验证损失、一个清晰显示 wall-clock-time 横轴小于 45 分钟的相关学习曲线，以及你做了什么的描述。我们期望一个排行榜提交至少要超过 5.0 损失的朴素基线。在这里提交到排行榜：`github.com/stanford-cs336/assignment1-basics-leaderboard`。

## 参考文献

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl

---

**脚注**

¹ 完整列表见 pytorch.org/docs/stable/nn.html#containers。

² 注意，原始的 BPE 公式 R. Sennrich et al. [3] 指定要包含一个词尾（end-of-word）token。我们在训练字节级 BPE 模型时不加词尾 token，因为所有字节（包括空白和标点）都已包含在模型词表中。由于我们显式地表示空格和标点，学到的 BPE 合并自然会反映这些词边界。

³ 关于 Unicode 替换字符的更多信息，见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

⁴ 值得注意的是，虽然 `einops` 有大量的支持，但 `einx` 没有那么多实战检验。如果你发现 `einx` 有任何局限或 bug，尽可以回退到用 `einops` 加上一些更普通的 PyTorch。

⁵ 作为一个进阶目标，尝试把 key、query 和 value 的投影合并成单个权重矩阵，这样你只需要一次矩阵乘法。

⁶ 注意，$o_i[k]$ 指向量 $o_i$ 中索引 $k$ 处的值。

⁷ 这对应于 $x_{i+1}$ 上的 Dirac delta 分布与预测的 softmax($o_i$) 分布之间的交叉熵。

⁸ 有时也常见使用一种让学习率回升（restart）的调度，以帮助越过局部最小值。

⁹ wandb.ai
