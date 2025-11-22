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
import json
import os
import time
import math
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
matplotlib.use("Agg")  
zh_font = FontProperties(fname="/usr/share/fonts/oprntype/noto/NotoSansCJK-Regular.ttc")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MODEL_DIR = "./checkpoints"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pt")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder.pt")

class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS",1:"EOS"}
        self.n_words = 2
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addword(word)

    def addword(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s) # 规范分解，合成字符=基础字符+附加符号
        if unicodedata.category(c) != 'Mn' # Mn 类：重音符号，Lu: 大写字母，Ll: 小写字母，Lm: 修饰符号，Lo: 其他符号
    )

def normalizeString(s):
    s = s.strip()  # 去除首尾空白字符
    s = s.lower()
    s = re.sub(r"[.。!！?？]", "", s)  # 去除标点符号
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s) 
    """
    ^：在方括号内表示 “取反”（匹配 “不包含以下字符” 的内容）；
    \w：匹配英文大小写字母（a-z、A-Z）、数字（0-9）、下划线（）；
    \u4e00-\u9fff：匹配所有中文汉字（Unicode 中中文汉字的编码范围）；
    +：匹配 “一个或多个连续” 的目标字符（避免单个噪声字符被拆成多个空格）。
    """
    s = re.sub(r"\s+"," ",s) 
    """
    \s+：匹配 一个或多个连续的空白字符（比如 （三个空格）、\t\n（制表符 + 换行符）
    """
    if ' ' not in s:
        s = ' '.join(list(s))
    return s

def readLangs(lang1,lang2,reverse=False):
    print("Reading lines...")
    file_path = "./data/eng-cmn.txt"
    with open(file_path,encoding="utf_8") as file:
        lines = file.readlines()

    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p))for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    return input_lang,output_lang,pairs

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
        len(p[1].split(' '))<MAX_LENGTH and \
            p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1,lang2,reverse=False):
    input_lang,output_lang,pairs = readLangs(lang1,lang2,reverse)
    print("Read %s sentence pairs" % len(pairs)) 
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name,input_lang.n_words)
    print(output_lang.name,output_lang.n_words)
    return input_lang,output_lang,pairs

input_lang,output_lang,pairs= prepareData('eng','cmn',True)
print(random.choice(pairs))

# 注意这里默认batch_size=1, 因为我们每次只传入一个句子,翻译逐词输入，所以这里的input_lengths和target_lengths都是1
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hiden_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hiden_size
        self.embedding = nn.Embedding(input_size,hiden_size)
        self.gru = nn.GRU(hiden_size,hiden_size)
    
    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output,hidden = self.gru(output,hidden)
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)
    

class AttnDecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p=0.1,max_length=MAX_LENGTH):
        super(AttnDecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length    
        
        self.embedding = nn.Embedding(self.output_size,self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2,self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size,self.hidden_size)
        self.out = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,input,hidden,encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0],hidden[0]),1)),dim = 1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0],attn_applied[0]),1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

def indexesFromSentence(lang,sentence):
    indexes = []
    for word in sentence.split(' '):
        word = word.strip()
        if not word:
            continue
        indexes.append(lang.word2index[word])
    
    if not indexes:
        indexes = [EOS_token]

    return indexes

def tensorFromSentence(lang,sentence):
    indexes = indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1) #(seq_len, batch_size)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang,pair[0])
    target_tensor = tensorFromSentence(output_lang,pair[1])
    return (input_tensor,target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0]
    
    decoder_input = torch.tensor([[SOS_token]],device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
            loss += criterion(decoder_output,target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
            topv,topi= decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() # 切断梯度传播
            loss += criterion(decoder_output,target_tensor[di])
            if decoder_input.item() == EOS_token:
                break   # 遇到EOS_token时停止翻译, 因为翻译结果已经生成完毕
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since,percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

learning_rate = 0.01
epoch = 1
batch_size = 1

def save_metrics(epoch,loss,filename = 'training_log.txt'):
    """Append a JSON object with epoch, loss, accuracy to filename."""
    record = {"epoch": epoch, "loss": loss}
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print('Failed to write metrics to', filename, ':', e)

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
    showPlot(plot_losses, save_path="train_loss.png")

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

def evaluate(encoder,decoder,sentence,max_langth=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang,sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_langth,encoder.hidden_size,device=device)
        for ei in range(input_length):
            encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0,0]
        
        decoder_input = torch.tensor([[SOS_token]],device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_langth,max_langth)
        for di in range(max_langth):
            decoder_output,decoder_hidden,decoder_attention = decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_attention[di] = decoder_attention.data
            topv,topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words,decoder_attentions[:di+1]


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
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000, learning_rate=learning_rate)
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
    save_path = f"attention_{index}.png"
    showAttention(norm_sentence, output_words, attentions, save_path=save_path)


evaluateAndShowAttention("他 和 他 的 邻 居 相 处 ", index=1)
evaluateAndShowAttention("我 肯 定 他 会 成 功 的 ", index=2)
evaluateAndShowAttention("他 總 是 忘 記 事 情", index=3)
evaluateAndShowAttention("我 们 非 常 需 要 食 物 ", index=4)
evaluateAndShowAttention("你 只 是 玩 ", index=5)










