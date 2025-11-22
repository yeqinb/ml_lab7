#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from io import open
import unicodedata
import string
import re
import random
import os
import time
import math
import json
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import nltk
from nltk.translate.bleu_score import sentence_bleu

# ========= 全局配置 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对齐你现在的设定
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10   # 你之前就是 10

# 为了结果更稳定，固定随机种子（可选，但强烈建议）
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_DIR = "./checkpoints2"
os.makedirs(MODEL_DIR, exist_ok=True)
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.pt")


# ========= 语言类 =========
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {
            PAD_token: "<PAD>",
            SOS_token: "<SOS>",
            EOS_token: "<EOS>",
        }
        self.word2count = {}
        self.n_words = 3  # PAD, SOS, EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            word = word.strip()
            if not word:
                continue
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# ========= 文本预处理（英文用，中文基本不动） =========
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s, lang="eng"):
    s = s.strip()
    if lang == "eng":
        s = unicodeToAscii(s.lower())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    else:
        # 中文这边你原来是空格分好词，这里基本不动，只去掉多余空格
        s = re.sub(r"\s+", " ", s).strip()
        return s


# ========= 数据读取 =========
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # 数据文件路径，按你之前的习惯
    data_path = os.path.join("data", "eng-cmn.txt")
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')

    pairs = []
    for l in lines:
        parts = l.split('\t')
        if len(parts) < 2:
            continue
        eng = normalizeString(parts[0], "eng")
        cmn = normalizeString(parts[1], "cmn")
        pairs.append([eng, cmn])

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    """控制句子长度：+1 是留给 EOS"""
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [p for p in pairs if filterPair(p)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %d sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %d sentence pairs" % len(pairs))
    print("Counting words...")
    for p in pairs:
        input_lang.addSentence(p[0])
        output_lang.addSentence(p[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# ========= 索引 & Tensor =========
def sentence_to_indices(lang, sentence):
    """句子 -> 索引列表（不含 PAD，结尾加 EOS）"""
    idxs = []
    for w in sentence.split(' '):
        w = w.strip()
        if not w:
            continue
        # 训练集里都会出现，这里 safe 一点，未知词就映射到 PAD
        idxs.append(lang.word2index.get(w, PAD_token))
    if len(idxs) == 0:
        idxs = [EOS_token]
    else:
        idxs.append(EOS_token)
    return idxs


def pad_sequences(seqs, max_len=None, pad_value=PAD_token):
    """seqs: list of list[int]  -> 长度对齐后的 tensor (batch, max_len)"""
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    padded = []
    lengths = []
    for s in seqs:
        l = len(s)
        lengths.append(l)
        if l < max_len:
            s = s + [pad_value] * (max_len - l)
        padded.append(s)
    return torch.tensor(padded, dtype=torch.long, device=device), lengths


def batch_from_pairs(pairs, input_lang, output_lang):
    """
    pairs: list of [input_str, target_str]
    返回:
      input_batch:  (batch, max_in_len)
      input_lengths: list[int]
      target_batch: (batch, max_out_len)
      target_lengths: list[int]
    """
    input_seqs = [sentence_to_indices(input_lang, p[0]) for p in pairs]
    target_seqs = [sentence_to_indices(output_lang, p[1]) for p in pairs]

    input_batch, input_lengths = pad_sequences(input_seqs, pad_value=PAD_token)
    target_batch, target_lengths = pad_sequences(target_seqs, pad_value=PAD_token)

    # pack_padded_sequence 要求按长度降序排
    input_lengths, perm_idx = torch.tensor(input_lengths).sort(0, descending=True)
    input_batch = input_batch[perm_idx]

    target_batch = target_batch[perm_idx]
    target_lengths = [target_lengths[i] for i in perm_idx]

    return input_batch, input_lengths.tolist(), target_batch, target_lengths


# ========= 模型 =========
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, input_batch, input_lengths, hidden=None):
        """
        input_batch: (batch, seq_len)
        input_lengths: list[int]
        """
        embedded = self.embedding(input_batch)  # (batch, seq_len, hidden)
        # 打包
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=True
        )
        outputs, hidden = self.gru(packed, hidden)  # outputs: PackedSequence
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )  # (batch, seq_len, hidden)

        return outputs, hidden  # outputs 用于 attention，hidden 传给 decoder


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout_p if n_layers > 1 else 0.0,
            batch_first=True,
        )
        # Luong dot-style: attn(h_t, H) -> weights; 然后拼接 context 和 h_t 再输出
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs, encoder_mask):
        """
        input_step: (batch,)  当前 step 的 token id
        last_hidden: (n_layers, batch, hidden)
        encoder_outputs: (batch, src_len, hidden)
        encoder_mask: (batch, src_len)  True=有效，False=PAD
        """
        embedded = self.embedding(input_step).unsqueeze(1)  # (batch, 1, hidden)
        embedded = self.dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)  # rnn_output: (batch, 1, hidden)
        rnn_output = rnn_output  # (batch, 1, hidden)

        # dot attention: score = h_t * H^T
        # rnn_output: (batch, 1, hidden)
        # encoder_outputs: (batch, src_len, hidden)
        attn_energies = torch.bmm(
            rnn_output, encoder_outputs.transpose(1, 2)
        )  # (batch, 1, src_len)

        # mask PAD positions
        # encoder_mask: (batch, src_len) -> (batch, 1, src_len)
        encoder_mask_exp = encoder_mask.unsqueeze(1)  # bool
        attn_energies = attn_energies.masked_fill(~encoder_mask_exp, -1e9)

        attn_weights = F.softmax(attn_energies, dim=-1)  # (batch, 1, src_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden)

        # Luong dot 原论文还会拼接 context 和 rnn_output 再过一层线性，
        # 这里我们简单一点，直接用 rnn_output 做输出层
        output = rnn_output.squeeze(1)  # (batch, hidden)
        output = self.out(output)       # (batch, output_size)
        output = F.log_softmax(output, dim=-1)
        return output, hidden, attn_weights.squeeze(1)  # attn_weights: (batch, src_len)


# ========= 训练 =========
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent + 1e-8)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train_batch(
    input_batch,
    input_lengths,
    target_batch,
    target_lengths,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    teacher_forcing_ratio=0.5,
):
    """
    一个 batch 的训练
    input_batch:   (batch, src_len)
    input_lengths: list[int]
    target_batch:  (batch, tgt_len)
    target_lengths:list[int]
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = input_batch.size(0)
    max_target_len = max(target_lengths)

    # encoder mask: (batch, src_len), True=非PAD
    encoder_mask = (input_batch != PAD_token)

    # 编码
    encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths)

    # 解码初始输入是 SOS
    decoder_input = torch.full(
        (batch_size,), SOS_token, dtype=torch.long, device=device
    )
    decoder_hidden = encoder_hidden  # 直接用 encoder 的最后 hidden

    # Teacher Forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    loss = 0.0
    n_tot_tokens = 0

    if use_teacher_forcing:
        # 把 target 的 token 逐步喂给 decoder
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_mask
            )
            # 真实的目标
            target_t = target_batch[:, t]  # (batch,)
            # 忽略 PAD
            loss_step = criterion(decoder_output, target_t)
            loss += loss_step
            # 统计非PAD个数，仅用于平均
            n_tot_tokens += (target_t != PAD_token).sum().item()
            # 下一个输入
            decoder_input = target_t
    else:
        # 不用 teacher forcing，使用自己的预测
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_mask
            )
            topv, topi = decoder_output.topk(1)
            next_input = topi.squeeze(1).detach()  # (batch,)
            target_t = target_batch[:, t]
            loss_step = criterion(decoder_output, target_t)
            loss += loss_step
            n_tot_tokens += (target_t != PAD_token).sum().item()
            decoder_input = next_input

    if n_tot_tokens == 0:
        # 极端情况（全是 PAD），基本不会出现
        avg_loss = 0.0
    else:
        avg_loss = loss.item() / n_tot_tokens

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return avg_loss

def save_metrics(epoch, loss, filename='training_log1.txt'):
    """Append a JSON object with epoch, loss, accuracy to filename."""
    record = {"epoch": epoch, "loss": loss}
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print('Failed to write metrics to', filename, ':', e)



def train_iters(
    encoder,
    decoder,
    n_iters,
    pairs,
    input_lang,
    output_lang,
    print_every=1000,
    learning_rate=0.01,
    batch_size=32,
    teacher_forcing_ratio=0.5,
):
    print("Starting training...")
    start = time.time()
    print_loss_total = 0

    encoder.train()
    decoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=PAD_token)

    for it in range(1, n_iters + 1):
        # 随机采样一个 batch
        batch_pairs = [random.choice(pairs) for _ in range(batch_size)]
        input_batch, input_lengths, target_batch, target_lengths = batch_from_pairs(
            batch_pairs, input_lang, output_lang
        )

        loss = train_batch(
            input_batch,
            input_lengths,
            target_batch,
            target_lengths,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        print_loss_total += loss
        save_metrics(it, loss)
        if it % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f"{time_since(start, it / n_iters)}  "
                f"Iter {it}/{n_iters}  Loss: {print_loss_avg:.4f}"
            )


# ========= 评估 =========
def tensor_from_sentence_for_eval(lang, sentence):
    idxs = sentence_to_indices(lang, sentence)
    return torch.tensor([idxs], dtype=torch.long, device=device), [len(idxs)]


def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_batch, input_lengths = tensor_from_sentence_for_eval(input_lang, sentence)
        # mask
        encoder_mask = (input_batch != PAD_token)

        encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths)
        decoder_hidden = encoder_hidden

        decoder_input = torch.tensor(
            [SOS_token], dtype=torch.long, device=device
        )  # (1,)

        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_mask
            )
            topv, topi = decoder_output.topk(1)
            ni = topi.item()
            if ni == EOS_token:
                decoded_words.append("<EOS>")
                break
            elif ni == PAD_token:
                # 解码阶段遇到 PAD 直接视为结束
                decoded_words.append("<PAD>")
                break
            else:
                decoded_words.append(output_lang.index2word.get(ni, "UNK"))
            decoder_input = torch.tensor([ni], dtype=torch.long, device=device)

        return decoded_words


def evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs, n=10):
    encoder.eval()
    decoder.eval()

    sum_bleu = 0.0

    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, input_lang, output_lang, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

        # 用 NLTK BLEU（和你之前的一样）
        ref = pair[1].strip().split(' ')
        ref.append('<EOS>')
        references = [ref]
        # hypothesis 里已经含有 <EOS>
        bleu = sentence_bleu(references, output_words)
        sum_bleu += bleu

    print("Average BLEU over %d samples: %.4f" % (n, sum_bleu / n))


# ========= 模型保存 / 加载 =========
def save_models(encoder, decoder):
    torch.save(encoder.state_dict(), ENCODER_PATH)
    torch.save(decoder.state_dict(), DECODER_PATH)
    print("Models saved.")


def load_models(encoder, decoder):
    if os.path.exists(ENCODER_PATH):
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
        print("Loaded encoder from", ENCODER_PATH)
    if os.path.exists(DECODER_PATH):
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
        print("Loaded decoder from", DECODER_PATH)




# ========= 主入口 =========
def main():
    nltk.download('punkt')

    input_lang, output_lang, pairs = prepareData("eng", "cmn", reverse=True)
    print("Example pair:", random.choice(pairs))

    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # 你可以调 batch_size 和 n_iters 看效果
    train_iters(
        encoder,
        decoder,
        n_iters=50000,
        pairs=pairs,
        input_lang=input_lang,
        output_lang=output_lang,
        print_every=5000,
        learning_rate=0.01,
        batch_size=1,              # <== 这里就是 batch 版本
        teacher_forcing_ratio=0.5,
    )

    save_models(encoder, decoder)

    # 随机评估若干句子
    evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs, n=100)

if __name__ == "__main__":
    main()
