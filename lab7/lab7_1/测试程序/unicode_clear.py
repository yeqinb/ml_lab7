import unicodedata

# 定义核心函数（保持原逻辑不变）
def unicodeToAscii(s):
    """将带重音/附加符号的Unicode字符，转换为无附加符号的纯ASCII字符"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'  # 过滤非间距标记（重音/音调符号）
    )

# ---------------------- 演示例子 ----------------------
if __name__ == "__main__":
    # 测试用例：覆盖西欧语言、特殊符号、中文（验证对中文无影响）
    test_cases = [
        # 1. 法语（带重音字母）
        "café",          # 咖啡
        "déjà vu",       # 似曾相识
        "àèìòùâêîôûäëïöü",  # 法语各类重音字母
        # 2. 西班牙语
        "niño",          # 男孩
        "señor",         # 先生
        # 3. 德语
        "cölner",        # 科隆的（形容词）
        "schön",         # 美丽的
        "straße",        # 街道（注意：ß 不是 Mn 类，会保留，ASCII中无对应，所以输出仍是 ß）
        # 4. 其他语言（葡萄牙语、意大利语）
        "pão",           # 葡萄牙语：面包
        "più",           # 意大利语：更多
        # 5. 中文（验证无影响）
        "中文测试：café 咖啡",
        # 6. 混合场景（字母+数字+符号）
        "123 - caféñöü 测试！@#",
    ]

    # 打印结果对比表格
    print("=" * 60)
    print(f"{'输入（带重音/特殊字符）':<30} | {'输出（纯ASCII/保留非重音）':<30}")
    print("=" * 60)

    for input_str in test_cases:
        output_str = unicodeToAscii(input_str)
        print(f"{input_str:<30} | {output_str:<30}")

    # ---------------------- 额外说明 ----------------------
    print("\n" + "-" * 60)
    print("关键说明：")
    print("函数只过滤 'Mn' 类字符（重音/音调等附加符号），其他字符（字母、数字、中文、标点）均保留")
    print("中文本身无 '基础字符+重音' 结构，所以处理后无变化")
    print("德语 'ß'、俄语字母等非ASCII字符，因不属于 'Mn' 类，会原样保留（ASCII中无对应基础字符）")