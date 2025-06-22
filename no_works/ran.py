import random
import string

def get_random_string(length: int = 10) -> str:
    """
    生成一个指定长度的随机字符串，要求第一个字符是大写字母，
    后面字符是大写字母、小写字母和数字的随机组合。

    Args:
        length (int, optional): 随机字符串长度，默认10。

    Returns:
        str: 生成的随机字符串。
    """
    if length < 1:
        return ''  # 长度小于1时返回空字符串

    first_char = random.choice(string.ascii_uppercase)  # 第一个字符大写字母
    other_chars = ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length - 1)
    )
    return first_char + other_chars

for _ in range(5):
    print(get_random_string(20))  # 测试输出一个10字符长的随机字符串

chunk='data: "我是"'
if chunk.strip():  
    # 移除"data: "前缀和两边的引号
    content = chunk.strip()
    if content.startswith('data: '):
        content = content[6:]  # 移除前缀
        print("content1:",content)
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]  # 移除引号
            print("content2:",content)

