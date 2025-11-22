lab1：完成基本任务(batch_size=1，50000轮）
lab2：padding+mask，允许动态的batch_size，（batch_size=1，50000轮）
lab3：batch_size=32

基本情况：

lab1和lab2情况基本相同，lab2的bleu_score稍高，lab1和lab2可以对比着来看

lab3作为一个反面例子：
batch_size太大 → 梯度越平滑 → 学到的模式越“平均化” → 翻译更保守，更不细腻
batch=1 时 BLEU 高，但是 batch=32 时 BLEU 会掉。
来自 NMT 和 Vision 的研究都证明：
大 batch = 更差的泛化
尤其是：
数据少（翻译数据只有 1471 pairs）
模型规模固定（256 hidden）
学习率不变（SGD + lr=0.01）
大 batch 会让训练走向“平滑但泛化差”的方向。
最终表现：
训练 loss 很低（看起来收敛很好）
测试翻译硬伤变多（BLEU 掉）