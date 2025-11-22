#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import jupyter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
matplotlib.use("Agg")  # 非交互后端，专门用来保存图片

import numpy as np

# 精确指定一个支持中文的字体
zh_font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


import nltk
from nltk.translate.bleu_score import sentence_bleu
import json
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型保存路径
MODEL_DIR = "./checkpoints1"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.pt")

# In[2]:
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
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


# 其中normalizeString函数中的正则表达式需对应更改，否则会将中文单词替换成空格
import re

def normalizeString(s: str) -> str:
    s = s.strip()
    s = s.lower()
    s = re.sub(r"[.。!！?？]", "", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if ' ' not in s:
        s = ' '.join(list(s))

    return s



# In[4]:


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    file_path = "./data/eng-cmn.txt"
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()
    

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]


    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# In[5]:


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# In[6]:


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



# In[7]:


input_lang, output_lang, pairs = prepareData('eng', 'cmn', True)
print(random.choice(pairs))


# In[8]:


'''file_path = "./data/eng-cmn.txt"
with open(file_path, encoding='utf-8') as file:
    lines = file.readlines()
pairs = [[normalizeString(l).split('\t')[:2]] for l in lines]
cn = []
eng = []
for p in pairs:
    p=np.array(p)
    eng.append([p[0,0]])
    cn.append([p[0,1]])'''


# # The Encoder

# In[9]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# # Attention Decoder

# In[10]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# # Preparing Training Data

# In[11]:


def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        word = word.strip()
        if not word:
            continue  # 跳过空 token（比如末尾多出来的空格）
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            # 未登录词可以选择跳过，这里直接忽略
            # 也可以在这里打印 warn 看看有哪些 OOV
            # print(f"[OOV] {word}")
            continue
    if not indexes:
        # 防御：至少有一个 EOS，避免后续再炸
        indexes = [EOS_token]
    return indexes



def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# # Training the model

# In[12]:


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# In[13]:


#This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[14]:


# jetML平台配置



learning_rate = 0.01
epoch = 1
batch_size = 1

def save_metrics(epoch, loss, filename='training_log.txt'):
    """Append a JSON object with epoch, loss, accuracy to filename."""
    record = {"epoch": epoch, "loss": loss}
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print('Failed to write metrics to', filename, ':', e)






# In[15]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # Save metrics to a text file (JSON per line). Use `iter` as epoch and placeholder accuracy=0.
        try:
            save_metrics(iter, loss)
        except NameError:
            # If save_metrics isn't available for some reason, skip writing.
            pass

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # 例如保存到当前目录
    showPlot(plot_losses, save_path="train_loss1.png")



# # Plotting results

# In[16]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points, save_path="train_loss.png"):
    # 创建画布和坐标轴
    fig, ax = plt.subplots()

    # 画曲线
    ax.plot(points)

    # y 轴刻度间隔
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")

    fig.tight_layout()

    # 保存到文件，不弹窗
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {save_path}")



# # Evaluation

# In[17]:


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# In[18]:


def evaluateRandomly(encoder, decoder, n=100):
    sum_scores = 0
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        w = []
        words = pair[1].strip(' ').split(' ')
        words.append('<EOS>')
        w.append(words)
        bleu_score = sentence_bleu(w, output_words)
        sum_scores += bleu_score
    print('The bleu_score is ', sum_scores/n)


# # Training and Evaluating

# In[19]:
def save_models(encoder, decoder,
                encoder_path=ENCODER_PATH,
                decoder_path=DECODER_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved encoder to {encoder_path}")
    print(f"Saved decoder to {decoder_path}")


def load_models(encoder, decoder,
                encoder_path=ENCODER_PATH,
                decoder_path=DECODER_PATH):
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    print(f"Loaded encoder from {encoder_path}")
    print(f"Loaded decoder from {decoder_path}")

hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# ==== 关键逻辑：如果已有模型文件，就直接加载，不再训练 ====
if os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH):
    print("===> Found saved models, loading for evaluation...")
    load_models(encoder1, attn_decoder1)
else:
    print("===> No saved models, start training...")
    # 第一次可以用 75000，调试阶段可以先小一点比如 5000
    trainIters(encoder1, attn_decoder1, 50000, print_every=5000, learning_rate=learning_rate)
    # 训练完保存下来
    save_models(encoder1, attn_decoder1)
# ============================================================

evaluateRandomly(encoder1, attn_decoder1)


def showAttention(input_sentence, output_words, attentions, save_path="attention.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # 准备标签
    input_tokens = input_sentence.split(' ') + ['<EOS>']
    output_tokens = output_words  # 已包含 <EOS>

    # 设置 ticks 对齐
    ax.set_xticks(range(len(input_tokens)))
    ax.set_yticks(range(len(output_tokens)))

    # 关键：这里显式指定 fontproperties=zh_font
    ax.set_xticklabels(input_tokens, rotation=90, fontproperties=zh_font)
    ax.set_yticklabels(output_tokens, fontproperties=zh_font)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Saved] {save_path}")




def evaluateAndShowAttention(input_sentence, index=0):
    norm_sentence = normalizeString(input_sentence)
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, norm_sentence)

    print('input =', norm_sentence)
    print('output =', ' '.join(output_words))

    # 每张图片保存不同文件
    save_path = f"attention1_{index}.png"
    showAttention(norm_sentence, output_words, attentions, save_path=save_path)




evaluateAndShowAttention("他 和 他 的 邻 居 相 处 ", index=1)
evaluateAndShowAttention("我 肯 定 他 会 成 功 的 ", index=2)
evaluateAndShowAttention("他 總 是 忘 記 事 情", index=3)
evaluateAndShowAttention("我 们 非 常 需 要 食 物 ", index=4)
evaluateAndShowAttention("你 只 是 玩 ", index=5)

