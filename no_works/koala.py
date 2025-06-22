"""
https://koala.sh/chat

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
base_model = "gpt-4o-mini"


data = [
    {"id": "gpt-4o-mini", "model": "gpt-4o-mini", "object": "model", "created": 1722400000000, "owned_by": "OpenAI", "type": "text"}

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
        model: str = None,
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
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'Accept': 'text/event-stream',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
        'Referer': 'https://koala.sh/chat',
        'Flag-Real-Time-Data': 'false',
        # 'Visitor-ID':  get_random_string(20),
        'Visitor-ID':'M104xcG4CrFMUerYxF05',
        "Origin": "https://koala.sh",
        "Alt-Used": "koala.sh",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "TE": "trailers",
        'Content-Type': 'application/json',
        # '_ga':'GA1.1.135562489.1750346767; crisp-client%2Fsession%2Fb21d4851-f776-4e10-bd26-bef59f54886b=session_089e7dd9-abf3-4030-9a18-471b53b0bd5c; __stripe_mid=38b1254a-4bf6-47b7-bea2-521320c2ec7eccf3f0; _ga_9LCF2TJ2CY=GS2.1.s1750346767$o1$g1$t1750347235$j56$l0$h0; cf_clearance=YYMxb4yyeVm1PbloBVjeBoLgVuWA_ZHKlhCocXqkelg-1750604658-1.2.1.1-6rSwpk64gwSlGGQ8B1Jz7uOhOaRmoJ7badfxHDZ45Oa5WSLMiVO9Og3NSHdR8vO7TxvPv7RnuBYlhQnga9uBBmPKHUr8yPvC9749fe87Ak626xn9BWDf8gWCoYC4bvPCibP.8UUoMjCr36VCeCD458XAnXwY2UdixKzLdEhZ7PjopyvGiEnHuE.eHB61T.FT0czkeNg5Gff1E5Ne7ciD_EpWY58FlflGxv1dvuz__e8VMh96zbqVMvyeOQ64rgW1QFVEbvZJmNCV.dyke76F.6MrdkfpLU.D518YEovmgwrrFZZHxcpGYBjtjiEvBy37_ZrEFxsKuvqN87zIUqK6PksWn3pz3ecCi1Fk8Fvp3RE; _iidt=ZNotD0v/7EryLDKgMfPJkHRUD0wkAF2z640m0j7Bg/Yhml3oFu1ZrI01GTDSk+zMKV0xEi93enabFTn9Oj23+KYdqPqHM3+1pmUDLac=; _vid_t=otB8AS0XrWc4IFyp/yLrAS7/4sA7Gx09/XvnpcS3YjoBf4ilhBCRXZqSmnVCP8wOREn1C3VgFK/x00Lqr0qXo+6C5L7lIX6xUQaJ8PM=; __stripe_sid=c2cc3f5c-38d0-474f-9ff2-2b7ad43aff1d0628cb; ph_phc_sUBgtFpFGfL4lIY24ZS4PcZTNaRvtHCCh3XdWQE29CO_posthog=%7B%22distinct_id%22%3A%2201979833-dae0-774d-afef-6677cf0db295%22%2C%22%24sesid%22%3A%5B1750605230767%2C%2201979833-f2af-709b-b7cb-502b837c22d8%22%2C1750605230767%5D%7D',
        'dnt':'1',
        'priority':'u=1, i',
        'sec-ch-ua':'"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        'sec-ch-ua-mobile':'?0',
        'sec-ch-ua-platform':'"Windows"'
    }

    data_proxy = {
        "input": user_prompt,
        "inputHistory": [],
        "outputHistory": [],
        "model": model
    }
    if debug:
        print(json.dumps(headers_proxy, indent=4))
        print(json.dumps(data_proxy, indent=4))
    return chat_completion(model=model, headers=headers_proxy, payload=data_proxy)



def parse_response(response_text):
    """
    逐行解析SSE流式响应并提取delta.content字段
    包含多层结构校验，确保安全访问嵌套字段
    返回标准API响应格式
    """
    lines = response_text.split('\n')
    result = ""
    created = None
    object_type = None
    
    for line in lines:
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                # 提取第一个data行的元信息
                if isinstance(data, dict) and not created:
                    created = data.get("created")
                    object_type = data.get("object")
                
                # 安全访问嵌套字段，确保是字典类型
                if isinstance(data, dict):
                    # 检查是否存在choices字段且为列表
                    if "choices" in data and isinstance(data["choices"], list):
                        for choice in data["choices"]:
                            # 检查每个choice是否为字典且包含delta字段
                            if isinstance(choice, dict) and "delta" in choice:
                                delta = choice["delta"]
                                # 确保delta是字典且包含content字段
                                if isinstance(delta, dict) and "content" in delta:
                                    content = delta["content"]
                                    # 确保content是字符串类型
                                    if isinstance(content, str):
                                        result += content
            except json.JSONDecodeError:
                continue
    
    # 计算token数量
    enc = tiktoken.get_encoding("cl100k_base")
    completion_tokens = len(enc.encode(result))
    
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
                "content": result
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }
    
    return response_data




def parse_response1(response_text):
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
    print(f"----: {response_text}")
    # 合并所有data块的内容
    result = ""
    lines = response_text.split('\n')
    for chunk in lines:
        if chunk.strip():  
            # 移除"data: "前缀和两边的引号
            content = chunk.strip()
            if content.startswith('data: '):
                content = content[6:]  # 移除前缀
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]  # 移除引号
                    # print("content2:", content)
            result += content
    
    # print(f"合并后的结果: {result}")
    created = None
    object_type = None

    # 计算token数量
    enc = tiktoken.get_encoding("cl100k_base")
    completion_tokens = len(enc.encode(result))
    
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
                "content": result
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }
    
    return response_data



def chat_completion(model, headers, payload):
    """处理用户请求并保留上下文"""
    try:
        url = "https://koala.sh/api/gpt/"
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
        return parse_response1(response.text)
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
