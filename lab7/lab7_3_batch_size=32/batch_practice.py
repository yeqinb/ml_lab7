from io import open
import unicodedata
import string
import re
import random
import json
import os
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth_fn = SmoothingFunction().method1


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker

# ========= 全局配置 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10

MODEL_DIR = "./checkpoints"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.pt")

zh_font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")  # 按你机器字体路径调整


# ========= 语言与数据预处理 =========
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(" "):
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
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = s.strip()
    s = s.lower()
    s = re.sub(r"[.。!！?？]", "", s)  # 去掉常见中英文句末标点
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    if " " not in s:
        # 全中文时拆成字
        s = " ".join(list(s))
    return s


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def readLangs(lang1, lang2, reverse=False, path="./data/eng-cmn.txt"):
    print("Reading lines...")
    with open(path, encoding="utf_8") as f:
        lines = f.readlines()

    # 只取前两列：英\t中
    pairs = [[normalizeString(s) for s in l.split("\t")[:2]] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    # 用词数过滤，保证不超过 MAX_LENGTH（注意这里是“<”，留一个位置给 EOS）
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH and p[1].startswith(eng_prefixes)


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


input_lang, output_lang, pairs = prepareData("eng", "cmn", True)
print("Example pair:", random.choice(pairs))


# ========= 索引 & Tensor =========
def indexesFromSentence(lang, sentence):
    idxs = []
    for word in sentence.split(" "):
        word = word.strip()
        if not word:
            continue
        idxs.append(lang.word2index[word])
    if not idxs:
        idxs = [EOS_token]
    return idxs


def tensorFromSentence(lang, sentence, max_length=MAX_LENGTH):
    """
    返回 shape: (max_length,)
    先加 EOS，再 padding；不过长，因为前面用 filterPairs 过滤了。
    """
    idxs = indexesFromSentence(lang, sentence)
    idxs.append(EOS_token)  # len(idxs) <= MAX_LENGTH

    while len(idxs) < max_length:
        idxs.append(PAD_token)

    return torch.tensor(idxs, dtype=torch.long, device=device)


def tensorsFromPair(pair):
    """
    pair: (input_sentence_str, target_sentence_str)
    使用全局 input_lang / output_lang
    """
    input_tensor = tensorFromSentence(input_lang, pair[0], MAX_LENGTH)
    target_tensor = tensorFromSentence(output_lang, pair[1], MAX_LENGTH)
    return input_tensor, target_tensor


# ========= 模型定义（与 batch_size 无关） =========
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # (seq_len, batch, input_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_step, hidden):
        """
        input_step: (batch_size,)  每个样本一个 token index
        hidden:     (1, batch_size, hidden_size)
        """
        batch_size = input_step.size(0)
        embedded = self.embedding(input_step)           # (batch_size, hidden_size)
        embedded = embedded.view(1, batch_size, -1)     # (1, batch_size, hidden_size)
        output, hidden = self.gru(embedded, hidden)     # output: (1, batch_size, hidden_size)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,
                 dropout_p=0.1, max_length=MAX_LENGTH):
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

    def forward(self, input_step, hidden, encoder_outputs):
        """
        input_step:      (batch_size,)
        hidden:          (1, batch_size, hidden_size)
        encoder_outputs: (max_length, batch_size, hidden_size)
        """
        batch_size = input_step.size(0)

        embedded = self.embedding(input_step)          # (batch_size, hidden_size)
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(0)               # (1, batch_size, hidden_size)

        # Attention
        attn_input = torch.cat((embedded[0], hidden[0]), dim=1)  # (batch_size, 2*hidden)
        attn_energies = self.attn(attn_input)                    # (batch_size, max_length)
        attn_weights = F.softmax(attn_energies, dim=1)           # (batch_size, max_length)

        # encoder_outputs: (L, B, H) -> (B, L, H)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        # (B, 1, L) x (B, L, H) -> (B, 1, H)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_applied = attn_applied.squeeze(1)                    # (batch_size, hidden_size)

        output = torch.cat((embedded[0], attn_applied), dim=1)    # (batch_size, 2*hidden)
        output = self.attn_combine(output)                        # (batch_size, hidden_size)
        output = output.unsqueeze(0)                              # (1, batch_size, hidden_size)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)                 # output: (1, batch_size, hidden)

        output = self.out(output[0])                              # (batch_size, output_size)
        output = F.log_softmax(output, dim=1)                     # (batch_size, output_size)
        return output, hidden, attn_weights


# ========= 训练相关 =========
teacher_forcing_ratio = 0.5


def random_batch(pairs, batch_size):
    """
    从 pairs 中随机抽 batch_size 个样本，返回:
      input_batch:  (MAX_LENGTH, batch_size)
      target_batch: (MAX_LENGTH, batch_size)
    """
    batch_pairs = random.sample(pairs, batch_size)

    input_seqs = []
    target_seqs = []
    for src, tgt in batch_pairs:
        input_seqs.append(tensorFromSentence(input_lang, src))   # (L,)
        target_seqs.append(tensorFromSentence(output_lang, tgt)) # (L,)

    # stack 成 (L, B)
    input_batch = torch.stack(input_seqs, dim=1)   # (MAX_LENGTH, batch_size)
    target_batch = torch.stack(target_seqs, dim=1) # (MAX_LENGTH, batch_size)

    return input_batch, target_batch


def train_step(input_batch, target_batch,
               encoder, decoder,
               encoder_optimizer, decoder_optimizer,
               criterion,
               max_length=MAX_LENGTH):
    """
    input_batch:  (seq_len, batch_size)
    target_batch: (seq_len, batch_size)
    """
    seq_len, batch_size = input_batch.size()
    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs = torch.zeros(
        max_length, batch_size, encoder.hidden_size, device=device
    )

    # ---------- Encoder ----------
    for ei in range(seq_len):
        input_step = input_batch[ei]        # (batch_size,)
        encoder_output, encoder_hidden = encoder(input_step, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

    # ---------- Decoder ----------
    decoder_input = torch.full(
        (batch_size,), SOS_token, dtype=torch.long, device=device
    )
    decoder_hidden = encoder_hidden

    use_teacher_forcing = (random.random() < teacher_forcing_ratio)
    loss = 0.0
    target_length = target_batch.size(0)

    # 统计“真正参与了 loss 计算的时间步数”
    effective_steps = 0

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            target_step = target_batch[di].view(-1)  # (batch_size,)

            # 如果这一整个时间步 target 全是 PAD，就跳过，避免 ignore_index 全忽略
            if (target_step != PAD_token).sum() == 0:
                continue

            cur_loss = criterion(decoder_output, target_step)
            if torch.isnan(cur_loss):
                # 保险起见防御一下
                continue

            loss += cur_loss
            effective_steps += 1

            decoder_input = target_step  # teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            target_step = target_batch[di].view(-1)

            if (target_step != PAD_token).sum() == 0:
                continue

            cur_loss = criterion(decoder_output, target_step)
            if torch.isnan(cur_loss):
                continue

            loss += cur_loss
            effective_steps += 1

            topv, topi = decoder_output.topk(1, dim=1)   # (batch_size, 1)
            decoder_input = topi.squeeze(1).detach()     # (batch_size,)

    if effective_steps == 0:
        # 非常极端的情况（比如全是空句子），直接返回 0 避免除以 0
        avg_loss = 0.0
    else:
        avg_loss = loss.item() / effective_steps

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return avg_loss



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def save_metrics(epoch, loss, filename="training_log.txt"):
    record = {"epoch": epoch, "loss": loss}
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Failed to write metrics:", e)


def showPlot(points, save_path="train_loss.png"):
    fig, ax = plt.subplots()
    ax.plot(points)

    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


def trainIters(encoder, decoder,
               n_iters,
               batch_size=32,
               print_every=1000,
               plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0.0
    plot_loss_total = 0.0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    for it in range(1, n_iters + 1):
        input_batch, target_batch = random_batch(pairs, batch_size)
        loss = train_step(input_batch, target_batch,
                          encoder, decoder,
                          encoder_optimizer, decoder_optimizer,
                          criterion)
        print_loss_total += loss
        plot_loss_total += loss

        try:
            save_metrics(it, loss)
        except Exception:
            pass

        if it % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0.0
            print(
                "%s (%d %d%%) %.4f"
                % (timeSince(start, it / n_iters), it, it / n_iters * 100, print_loss_avg)
            )

        if it % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0.0

    showPlot(plot_losses, save_path="train_loss.png")


# ========= 推理 & 可视化 =========
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        # 不需要 padding，只用真实长度 + EOS
        idxs = indexesFromSentence(input_lang, sentence)
        idxs.append(EOS_token)
        input_length = len(idxs)

        input_tensor = torch.tensor(idxs, dtype=torch.long, device=device)  # (L,)
        encoder_hidden = encoder.initHidden(batch_size=1)

        encoder_outputs = torch.zeros(
            max_length, 1, encoder.hidden_size, device=device
        )

        # Encoder
        for ei in range(input_length):
            input_step = input_tensor[ei].unsqueeze(0)  # (1,)
            encoder_output, encoder_hidden = encoder(input_step, encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]      # (1, hidden) -> [ei,0,:]

        # Decoder
        decoder_input = torch.tensor([SOS_token], device=device)  # (1,)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, attn_weights = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # attn_weights: (1, max_length)
            decoder_attentions[di, : attn_weights.size(1)] = attn_weights[0].cpu()

            topv, topi = decoder_output.data.topk(1, dim=1)  # (1,1)
            next_token = topi[0, 0].item()

            if next_token == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[next_token])

            decoder_input = topi.squeeze(1).detach()  # (1,)

        return decoded_words, decoder_attentions[: di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    sum_scores = 0.0
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")

        words = pair[1].strip(" ").split(" ")
        words.append("<EOS>")
        reference = [words]
        bleu_score = sentence_bleu(reference, output_words, smoothing_function=smooth_fn)
        sum_scores += bleu_score

    print("The bleu_score is ", sum_scores / n)


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


def showAttention(input_sentence, output_words, attentions, save_path="attention.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap="bone")
    fig.colorbar(cax)

    input_tokens = input_sentence.split(" ") + ["<EOS>"]
    output_tokens = output_words

    ax.set_xticks(range(len(input_tokens)))
    ax.set_yticks(range(len(output_tokens)))

    ax.set_xticklabels(input_tokens, rotation=90, fontproperties=zh_font)
    ax.set_yticklabels(output_tokens, fontproperties=zh_font)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def evaluateAndShowAttention(encoder, decoder, input_sentence, index=0):
    norm_sentence = normalizeString(input_sentence)
    output_words, attentions = evaluate(encoder, decoder, norm_sentence)

    print("input =", norm_sentence)
    print("output =", " ".join(output_words))

    save_path = f"attention_{index}.png"
    showAttention(norm_sentence, output_words, attentions, save_path=save_path)


# ========= 主流程 =========
hidden_size = 256
train_batch_size = 32
learning_rate = 0.01

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,dropout_p=0.1).to(device)

if os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH):
    print("===> Found saved models, loading for evaluation...")
    load_models(encoder1, attn_decoder1)
else:
    print("===> No saved models, start training...")
    # 调试可先用小一点的 n_iters，比如 5000
    trainIters(encoder1, attn_decoder1,
               n_iters=75000,
               batch_size=train_batch_size,
               print_every=5000,
               learning_rate=learning_rate)
    save_models(encoder1, attn_decoder1)

# 随机评估
evaluateRandomly(encoder1, attn_decoder1)

# 画注意力图
evaluateAndShowAttention(encoder1, attn_decoder1, "他 和 他 的 邻 居 相 处 ", index=1)
evaluateAndShowAttention(encoder1, attn_decoder1, "我 肯 定 他 会 成 功 的 ", index=2)
evaluateAndShowAttention(encoder1, attn_decoder1, "他 總 是 忘 記 事 情", index=3)
evaluateAndShowAttention(encoder1, attn_decoder1, "我 们 非 常 需 要 食 物 ", index=4)
evaluateAndShowAttention(encoder1, attn_decoder1, "你 只 是 玩 ", index=5)
