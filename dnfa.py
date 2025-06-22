"""
https://chat9.yqcloud.top

update time: 2025.06.22
verson: 1.0.0
"""
import json
import re
import time
from datetime import datetime, timedelta
from typing import Set, Optional, List, Dict
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
import random
import string
# 禁用 SSL 警告
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)
urllib3.disable_warnings()

# 新增导入
import tiktoken
from datetime import datetime

debug = True

# 全局变量
last_request_time = 0  # 上次请求的时间戳
cache_duration = 14400  # 缓存有效期，单位：秒 (4小时)

'''用于存储缓存的模型数据'''
cached_models = {
    "object": "list",
    "data": [],
    "version": "1.0.0",
    "provider": "dnfa",
    "name": "dnfa",
    "default_locale": "en-US",
    "status": True,
    "time": 20250622
}



# 全局变量：存储所有模型的统计信息
# 格式：{model_name: {"calls": 调用次数, "fails": 失败次数, "last_fail": 最后失败时间}}
MODEL_STATS: Dict[str, Dict] = {}

def record_call(model_name: str, success: bool = True) -> None:
    """
    记录模型调用情况
    Args:
        model_name: 模型名称
        success: 调用是否成功
    """
    global MODEL_STATS
    if model_name not in MODEL_STATS:
        MODEL_STATS[model_name] = {"calls": 0, "fails": 0, "last_fail": None}

    stats = MODEL_STATS[model_name]
    stats["calls"] += 1
    if not success:
        stats["fails"] += 1
        stats["last_fail"] = datetime.now()



'''基础模型'''
base_model = "gpt-4"


data = [
    {"id": "gpt-4", "model": "gpt-4", "object": "model", "created": 1722400000000, "owned_by": "OpenAI", "type": "text"}

]

def get_models():
    """模型值"""
    models = {
        "object": "list",
        "data": data
    }
    return json.dumps(models)

def get_auto_model(cooldown_seconds: int = 300) -> str:
    """异步获取最优模型"""
    return base_model



def reload_check():
    """
    检查并更新系统状态
    """
    pass




def is_chatgpt_format(data):
    """Check if the data is in the expected ChatGPT format"""
    try:
        # If the data is a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return False  # If the string can't be parsed, it's not in the expected format

        # Now check if data is a dictionary and contains the necessary structure
        if isinstance(data, dict):
            # Ensure 'choices' is a list and the first item has a 'message' field
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                if "message" in data["choices"][0]:
                    return True
    except Exception as e:
        print(f"Error checking ChatGPT format: {e}")

    return False

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

def chat_completion_message(
        user_prompt,
        user_id: str = None,
        session_id: str = None,
        system_prompt="You are a helpful assistant.",
        model: str = 'gpt-4',
        stream=True,
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0):
    # 确保model有效
    if not model or model == "auto":
        model = get_auto_model()
    
    if debug:
        print(f"校准后的model: {model}")


    headers_proxy = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://chat9.yqcloud.top",
            "referer": "https://chat9.yqcloud.top/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        
    userId = f"#/chat/{int(time.time() * 1000)}"
    data_proxy = {
                "prompt": user_prompt,
                "userId": userId,
                "network": True,
                "system": system_prompt,
                "withoutContext": False,
                "stream": stream
            }
    if debug:
        print(json.dumps(headers_proxy, indent=4))
        print(json.dumps(data_proxy, indent=4))
    return chat_completion(model=model, headers=headers_proxy, payload=data_proxy)




def parse_response(response_text):
    """
    解析纯文本格式
    data: ""
    data: "我是"
    data: "一个"
    data: "基"
    data: "于"
    data: "人工"

    逐行解析SSE流式响应并提取delta.content字段
    包含多层结构校验，确保安全访问嵌套字段
    返回标准API响应格式
    """
    created = None
    object_type = None

    # 计算token数量
    enc = tiktoken.get_encoding("cl100k_base")
    completion_tokens = len(enc.encode(response_text))
    
    # 组装标准响应数据
    response_data = {
        "id": f"chatcmpl-{datetime.now().timestamp()}",
        "object": object_type or "chat.completion",
        "created": created or int(datetime.now().timestamp()),
        "model": "gpt-4o",  # 可根据需求调整来源
        "usage": {
            "prompt_tokens": 0,  # 需要根据实际prompt内容计算
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens
        },
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }
    
    return response_data



def chat_completion(model, headers, payload):
    """处理用户请求并保留上下文"""
    try:
        url = "https://api.binjie.fun/api/generateStream"
        if debug:
            print(f"url: {url}")
        response = requests.post(url=url, headers=headers, json=payload, verify=False, timeout=100)
        response.encoding = 'utf-8'
        response.raise_for_status()
        if response.status_code != 200:
            record_call(model, False)
        else:
            record_call(model, True)

        if debug:
            print(response.status_code)
            # print(response.text)
        # return response.json()
        # if stream:
        #     if debug:
        #         print('-------------this is streaming,will parse--------')
        #     return parse_response(response.text)
        # return response.text\

        return response.text
    except requests.exceptions.RequestException as e:
        record_call(model, False)
        if debug:
            print(f"请求失败,请检查网络或参数配置。: {e}")
    except (KeyError, IndexError) as e:
        record_call(model, False)
        if debug:
            print(f"解析响应时出错,解析响应内容失败。: {e}")
        return ""
    record_call(model, False)
    return {}


if __name__ == '__main__':
    # get_from_js_v3()
    # print("get_models: ", get_models())
    # print("cached_models:", cached_models)
    # print("base_url: ", base_url)
    # print("MODEL_STATS:", MODEL_STATS)
    # print("base_model:",base_model)
    # base_model = "QwQ-32B"

    result = chat_completion_message(user_prompt="你是什么模型？", model=base_model,stream=True)
    print(result)


    # base_model="Llama-4-Scout-Instruct"
    # result = chat_completion_message(user_prompt="你是什么模型？", model=base_model, stream=False)
    # print(result)




    # # 单次对话
    # result1 = chat_completion_message(
    #     user_prompt="你好，请介绍下你自己",
    #     # model=base_model,
    #     temperature=0.3
    # )
    # print(result1)

    # # 多轮对话
    # messages = [
    #     {"role": "system", "content": "你是一个助手"},
    #     {"role": "user", "content": "你好"}
    # ]
    # result2 = chat_completion_messages(messages)
    # print(result2)

 #    msg="""
 #    json 格式化
 # {"object": "list", "data": [{"id": "Qwen2.5-VL-72B-Instruct", "object": "model", "model": "Qwen2.5-VL-72B-Instruct", "created": 1744090984000, "owned_by": "Qwen2.5", "name": "Qwen o1", "description": "Deep thinking,mathematical and writing abilities \u2248 o3, taking photos to solve math problems", "support": "image", "tip": "Qwen o1"}, {"id": "DeepSeek-R1", "object": "model", "model": "DeepSeek-R1", "created": 1744090984000, "owned_by": "DeepSeek", "name": "DeepSeek R1", "description": "Deep thinking,mathematical and writing abilities \u2248 o3", "support": "text", "tip": "DeepSeek R1"}, {"id": "Llama3.3-70B", "object": "model", "model": "Llama3.3-70B", "created": 1744090984000, "owned_by": "Llama3.3", "name": "Llama3.3", "description": "Suitable for most tasks", "support": "text", "tip": "Llama3.3"}], "version": "0.1.125", "provider": "DeGPT", "name": "DeGPT", "default_locale": "en-US", "status": true, "time": 0}
 #    """
 #    ress = chat_completion_message(user_prompt=msg)
 #    print(ress)
 #    print(type(ress))
 #    print("\r\n----------\r\n\r\n")
