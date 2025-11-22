import re
import unicodedata
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 50


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            word = word.strip()
            if not word:
                continue
            self.addword(word)

    def addword(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = s.strip()                      # 去掉首尾空白
    s = s.lower()
    s = re.sub(r"[.。!！?？]", "", s)   # 去掉中英文句号、感叹号、问号
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    # 只保留英文/数字/下划线 + 中文，其它换成空格
    s = re.sub(r"\s+", " ", s)         # 多个空白合并为一个空格
    if ' ' not in s:
        # 全是连续中文时，拆成“字 空格 字 空格 …”
        s = ' '.join(list(s))
    return s


def readLangs(lang1, lang2, reverse=False, path="./data/eng-cmn.txt"):
    print("Reading lines...")
    with open(path, encoding="utf_8") as f:
        lines = f.readlines()

    # 只取前两个 tab 分隔字段（英\t中）
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    # 注意：这里只是按“词数”过滤，保证 < MAX_LENGTH
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False, path="./data/eng-cmn.txt"):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for p in pairs:
        input_lang.addSentence(p[0])
        output_lang.addSentence(p[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# ========= 索引 & tensor 部分 =========

def indexesFromSentence(lang, sentence):
    idxs = []
    for word in sentence.split(' '):
        word = word.strip()
        if not word:
            continue
        # 这里假设：经过 prepareData，所有训练用句子里的词都已经在词表里
        idxs.append(lang.word2index[word])
    if not idxs:
        idxs = [EOS_token]
    return idxs


def tensorFromSentence(lang, sentence, max_length=MAX_LENGTH):
    """
    返回 shape: (seq_len, batch_size) = (MAX_LENGTH, 1)
    只做 padding，不做截断，因为 filterPairs 已经过滤掉过长样本
    """
    idxs = indexesFromSentence(lang, sentence)
    idxs.append(EOS_token)            # 加 EOS，此时 len(idxs) <= MAX_LENGTH

    # 只 padding 到 MAX_LENGTH
    while len(idxs) < max_length:
        idxs.append(PAD_token)

    return torch.tensor(idxs, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    """
    pair: (input_sentence_str, target_sentence_str)
    使用全局的 input_lang / output_lang
    """
    input_tensor = tensorFromSentence(input_lang, pair[0], MAX_LENGTH)
    target_tensor = tensorFromSentence(output_lang, pair[1], MAX_LENGTH)
    return input_tensor, target_tensor


# ========= 示例 =========

input_lang, output_lang, pairs = prepareData('eng', 'cmn', True)
print("Example pair:", random.choice(pairs))

example_pair = random.choice(pairs)
inp, tgt = tensorsFromPair(example_pair)
print("input_tensor shape:", inp.shape)   # (MAX_LENGTH, 1)
print("target_tensor shape:", tgt.shape)  # (MAX_LENGTH, 1)
