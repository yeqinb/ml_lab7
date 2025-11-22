import torch
import torch.nn as nn

# 设定设备（CPU/GPU 均可）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):  # 修正原拼写错误 hiden_size→hidden_size
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # GRU 隐藏层维度
        self.embedding = nn.Embedding(input_size, hidden_size)  # 词嵌入层：词汇表大小→隐藏层维度
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU 层：输入/输出维度=隐藏层维度

    def forward(self, input, hidden):
        # input 维度：(batch_size,) → 这里 batch_size=2，输入是 2 个样本的词索引（如 [3, 7]）
        batch_size = input.size(0)  # 动态获取批量大小（适配任意 batch_size，不只限于 2）
        
        # 1. 词嵌入：(batch_size,) → (batch_size, hidden_size)
        embedded = self.embedding(input)
        print(f"嵌入后维度：{embedded.shape} → (batch_size, hidden_size)")
        
        # 2. 调整维度：适配 GRU 输入格式 (seq_len, batch_size, hidden_size)
        # seq_len=1（逐词输入），batch_size=2，hidden_size=设定值
        embedded = embedded.view(1, batch_size, -1)
        print(f"调整后维度：{embedded.shape} → (seq_len, batch_size, hidden_size)")
        
        # 3. GRU 前向传播：输入 (1,2,hidden_size)，隐藏层 (1,2,hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):  # 接收 batch_size 参数，动态初始化隐藏层
        # 隐藏层维度：(num_layers×num_directions, batch_size, hidden_size)
        # 单层单向 GRU → 第一维=1，batch_size=2，维度为 (1, 2, hidden_size)
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

# ---------------------- 验证 batch_size=2 的运行效果 ----------------------
if __name__ == "__main__":
    # 超参数设定
    input_size = 10  # 输入词汇表大小（假设有 10 个不同的词）
    hidden_size = 8   # GRU 隐藏层维度（嵌入向量维度同步为 8）
    batch_size = 2    # 目标批量大小=2

    # 1. 初始化编码器
    encoder = EncoderRNN(input_size, hidden_size).to(device)
    print(f"编码器初始化完成（batch_size={batch_size}）\n")

    # 2. 构造批量输入：2 个样本的词索引（如词汇表中第 3 个和第 7 个词）
    # 输入维度：(batch_size,) → (2,)
    input_batch = torch.tensor([3, 7], device=device)
    print(f"输入词索引：{input_batch}，输入维度：{input_batch.shape} → (batch_size,)")

    # 3. 初始化隐藏层（批量大小=2）
    hidden = encoder.initHidden(batch_size)
    print(f"初始隐藏层维度：{hidden.shape} → (1, batch_size, hidden_size)\n")

    # 4. 前向传播（编码过程）
    output, final_hidden = encoder(input_batch, hidden)

    # 5. 输出结果验证
    print(f"\nGRU 输出维度：{output.shape} → (seq_len, batch_size, hidden_size)")
    print(f"最终隐藏层维度：{final_hidden.shape} → (1, batch_size, hidden_size)")