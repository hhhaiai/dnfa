"""
https://deepinfra.com/

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
import aiohttp
import requests
# 禁用 SSL 警告
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)
urllib3.disable_warnings()

# 新增导入
import tiktoken
from datetime import datetime

debug = False
# 测试更多模型
testMoreModels = False
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
base_model = "deepseek-ai/DeepSeek-V3-0324"


data = [
    # deepseek
    {"id": "deepseek-ai/DeepSeek-V3", "model": "deepseek-ai/DeepSeek-V3", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    {"id": "deepseek-ai/DeepSeek-V3-0324", "model": "deepseek-ai/DeepSeek-V3-0324", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    {"id": "deepseek-ai/DeepSeek-R1-0528", "model": "deepseek-ai/DeepSeek-R1-0528", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    {"id": "deepseek-ai/DeepSeek-Prover-V2-671B", "model": "deepseek-ai/DeepSeek-Prover-V2-671B", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    {"id": "deepseek-ai/DeepSeek-R1", "model": "deepseek-ai/DeepSeek-R1", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    {"id": "deepseek-ai/DeepSeek-R1-Turbo", "model": "deepseek-ai/DeepSeek-R1-Turbo", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},
    # {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "object": "model", "created": 1722400000000, "owned_by": "DeepSeek", "type": "text"},


    # Qwen
    {"id": "Qwen/Qwen3-235B-A22B", "model": "Qwen/Qwen3-235B-A22B", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},
    {"id": "Qwen/Qwen3-30B-A3B", "model": "Qwen/Qwen3-30B-A3B", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},
    {"id": "Qwen/Qwen3-32B", "model": "Qwen/Qwen3-32B", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},
    {"id": "Qwen/Qwen3-14B", "model": "Qwen/Qwen3-14B", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},
    {"id": "Qwen/QwQ-32B", "model": "Qwen/QwQ-32B", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},
    # {"id": "Qwen/Qwen2.5-72B-Instruct", "model": "Qwen/Qwen2.5-72B-Instruct", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},
    # {"id": "Qwen/Qwen2-72B-Instruct", "model": "Qwen/Qwen2-72B-Instruct", "object": "model", "created": 1722400000000, "owned_by": "Qwen", "type": "text"},

    # llama
    {"id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "object": "model", "created": 1722400000000, "owned_by": "Meta", "type": "text"},
    {"id": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "object": "model", "created": 1722400000000, "owned_by": "Meta", "type": "text"},
    {"id": "meta-llama/Llama-Guard-4-12B", "model": "meta-llama/Llama-Guard-4-12B", "object": "model", "created": 1750602935401, "owned_by": "deepinfra", "type": "text"},
    # {"id": "microsoft/meta-llama/Llama-Guard-4-12B", "model": "microsoft/meta-llama/Llama-Guard-4-12B", "object": "model", "created": 1722400000000, "owned_by": "Microsoft", "type": "text"},
    # {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "object": "model", "created": 1722400000000, "owned_by": "Meta", "type": "text"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "object": "model", "created": 1722400000000, "owned_by": "Meta", "type": "text"},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "model": "meta-llama/Llama-3.3-70B-Instruct", "object": "model", "created": 1750602935402, "owned_by": "deepinfra", "type": "text"},
    # {"id": "cognitivecomputations/dolphin-2.9.1-llama-3-70b", "model": "cognitivecomputations/dolphin-2.9.1-llama-3-70b", "object": "model", "created": 1722400000000, "owned_by": "Cognitive Computations", "type": "text"},

    # phi4
    {"id": "microsoft/phi-4-reasoning-plus", "model": "microsoft/phi-4-reasoning-plus", "object": "model", "created": 1722400000000, "owned_by": "Microsoft", "type": "text"},
    {"id": "microsoft/phi-4", "model": "microsoft/phi-4", "object": "model", "created": 1722400000000, "owned_by": "Microsoft", "type": "text"},
    {"id": "microsoft/Phi-4-multimodal-instruct", "model": "microsoft/Phi-4-multimodal-instruct", "object": "model", "created": 1750602935414, "owned_by": "deepinfra", "type": "text"},


    # gemma
    {"id": "google/gemma-3-27b-it", "model": "google/gemma-3-27b-it", "object": "model", "created": 1722400000000, "owned_by": "Google", "type": "text"},
    {"id": "google/gemma-2-27b-it", "model": "google/gemma-2-27b-it", "object": "model", "created": 1750602935403, "owned_by": "deepinfra", "type": "text"},
    {"id": "google/gemma-3-12b-it", "model": "google/gemma-3-12b-it", "object": "model", "created": 1750602935407, "owned_by": "deepinfra", "type": "text"}, 
    {"id": "google/gemma-1.1-7b-it", "model": "google/gemma-1.1-7b-it", "object": "model", "created": 1750602935409, "owned_by": "deepinfra", "type": "text"}, 
    {"id": "google/gemma-3-4b-it", "model": "google/gemma-3-4b-it", "object": "model", "created": 1750602935414, "owned_by": "deepinfra", "type": "text"}, 
    {"id": "google/gemma-2-9b-it", "model": "google/gemma-2-9b-it", "object": "model", "created": 1750602935414, "owned_by": "deepinfra", "type": "text"}, 
    # {"id": "google/gemma-3-12b-it", "model": "google/gemma-3-12b-it", "object": "model", "created": 1722400000000, "owned_by": "Google", "type": "text"},

    # codegemma
    {"id": "google/codegemma-7b-it", "model": "google/codegemma-7b-it", "object": "model", "created": 1750602935393, "owned_by": "deepinfra", "type": "text"},

    # mixtral
    {"id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "object": "model", "created": 1750602935405, "owned_by": "deepinfra", "type": "text"},
    # {"id": "mistralai/Mistral-Small-24B-Instruct-2501", "model": "mistralai/Mistral-Small-24B-Instruct-2501", "object": "model", "created": 1722400000000, "owned_by": "Mistral AI", "type": "text"},
    # {"id": "cognitivecomputations/dolphin-2.6-mixtral-8x7b", "model": "cognitivecomputations/dolphin-2.6-mixtral-8x7b", "object": "model", "created": 1722400000000, "owned_by": "Cognitive Computations", "type": "text"},
    # {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "object": "model", "created": 1722400000000, "owned_by": "Mistral AI", "type": "text"},

    # WizardLM
    # {"id": "microsoft/WizardLM-2-8x22B", "model": "microsoft/WizardLM-2-8x22B", "object": "model", "created": 1722400000000, "owned_by": "Microsoft", "type": "text"},
    {"id": "microsoft/WizardLM-2-7B", "model": "microsoft/WizardLM-2-7B", "object": "model", "created": 1722400000000, "owned_by": "Microsoft", "type": "text"},

    # airoboros
    {"id": "deepinfra/airoboros-70b", "model": "deepinfra/airoboros-70b", "object": "model", "created": 1722400000000, "owned_by": "DeepInfra", "type": "text"},
    
     #lzlv
    {"id": "lizpreciatior/lzlv_70b_fp16_hf", "model": "lizpreciatior/lzlv_70b_fp16_hf", "object": "model", "created": 1722400000000, "owned_by": "Lizpreciatior", "type": "text"}
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
    model_map = {}  # model_id -> model_info
    all_models = []
    for item in data:
        # 获取模型ID，如果不存在则使用"id"字段
        model_id = item.get('model') or item['id']
        if model_id not in all_models:
            all_models.append(model_id)
        # 添加到映射字典，使用model作为键
        model_map[model_id] = item


    if testMoreModels:
        url = 'https://api.deepinfra.com/v1/openai/models'
        response = requests.get(url, verify=False, timeout=10)
        response.encoding = 'utf-8'
        response.raise_for_status()
        # if debug:
        #     print(response.status_code)
        #     print(response.text)
        
        if response.status_code == 200:
            api_models = response.json().get('data', [])
        
            for model in  api_models:
                __model_id = model['id']
                if debug:
                    print(__model_id)
                if __model_id not in all_models:
                    all_models.append(__model_id)
                    # 获取当前时间的毫秒时间戳
                    current_millis = int(time.time() * 1000)
                    # 如果模型ID以"VL"开头，则认为是图像模型，否则是文本模型
                    model_type = "image" if "VL" in model_id else "text"
                    cm_model = {
                        "id": __model_id,
                            "model": model["id"],
                            "object": model["object"],
                            "created": current_millis,
                            "owned_by": model.get("owned_by", "unknown"),
                            "type": model_type
                    }
                    if __model_id not in model_map:
                        model_map[__model_id] = cm_model
                        if debug:
                            print(f"新增模型: {__model_id}")
                    else:
                            if debug:
                                print(f"模型已存在: {__model_id}")
            if debug:
                print("============id个数:  ", len(all_models))
        else:
            if debug:
                print('网络请求失败，请求状态值:', response.status_code)
    
    print("all_models: ",all_models)
    if not all_models:
        print("没有获取到有效的模型ID，无法更新缓存。")
    else:
        for _model in all_models:
            if debug:
                print(f"正在测试模型: {_model}")
            result = chat_completion_message(user_prompt="say this is a test.", model=_model,stream=True)
            print(f"模型 {_model} 测试结果: {result}")
            if is_chatgpt_format(result):
                # if debug:
                    # print("success model: ", _model)
                print("success model: ", _model)
                # 如果模型不在data中，则添加
                if not any(d["model"] == _model for d in data):\
                    data.append(model_map[_model])
            else:
                if debug:
                    print("测试模型失败: ", _model)
    if debug:
        print("更新后的模型列表,现在模型个数: ", len(data))




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
    """未来会增加回话隔离: 单人对话,单次会话"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return chat_completion_messages(messages, user_id=user_id, session_id=session_id,
                                    model=model,  stream=stream, temperature=temperature,
                                    max_tokens=max_tokens, top_p=top_p, frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty)
def chat_completion_messages(
        messages,
        stream=True,
        model: str = None,
        user_id: str = None,
        session_id: str = None,
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
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'DNT': '1',
        'Origin': 'https://deepinfra.com',
        'Referer': 'https://deepinfra.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
        'X-Deepinfra-Source': 'web-page',
        'Accept': 'text/event-stream',
        'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': 'Windows',
    
    }

    data_proxy = {
        "messages": messages,
        "stream": stream,
        "model": model,
        "stream_options": {
            "include_usage":True,
            "continuous_usage_stats":True,
        }

    }
    if debug:
        print(json.dumps(headers_proxy, indent=4))
        print(json.dumps(data_proxy, indent=4))
    return chat_completion(model=model, headers=headers_proxy, payload=data_proxy,stream=stream)



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

def chat_completion(model, headers, payload,stream):
    """处理用户请求并保留上下文"""
    try:
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
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
        if stream:
            if debug:
                print('-------------this is streaming,will parse--------')
            return parse_response(response.text)
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

    reload_check()

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
