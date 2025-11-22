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


def main():
    tests = [
        "HelloWorld",
        "Hello World!",
        "你好世界！",
        "我爱这个世界。",
        "I love this world! 我爱这个世界！",
        "中英MixedSentence123!@#",
    ]

    for idx, sent in enumerate(tests, 1):
        norm = normalizeString(sent)
        print(f"样例 {idx}:")
        print(f"  原始: {sent}")
        print(f"  处理: {norm}")
        print("-" * 40)


if __name__ == "__main__":
    main()