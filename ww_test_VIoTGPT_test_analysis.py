# coding: utf-8
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import gc
import gradio as gr
import sys
import cv2
gr.close_all()
from shutil import copyfile
import torch
import numpy as np
import argparse
import inspect
import json
import jsonlines
import re
import uuid
from PIL import Image
import time
from VIoTGPT_Vision_nodemo1 import (
    PersonReid,
    VehicleReid,
    FSDetect,
    # ObjectLOCTool,
    # ImageSegmenter,
    HumanPoseEstimation,
    GroundingDINOWeaponDetector,
    VideoAnomalyDetection,
    CrowdCounting,
    HumanPoseTracking,
    FaceDetection,
    # LowLightEnhancer,
    HumanFallDetection,
    SceneRecognition,
    ViolenceDetection,
    PlateRecognition,
    GaitRecognition
    

)
from VIoTGPT_Vision_nodemodata import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX

from langchain.agents.initialize import initialize_agent

# 导入 AgentAction 用于类型检查
from langchain.schema import AgentAction

from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
sys.path.append("./main_models/")
from llama_model import LlamaModel
from lora_model import LoraModel
from main_models.qwen25VL_model import Qwen25VLModel
from main_models.qwen3VL_model import Qwen3VLModel
from main_models.llama3_model import Llama3Model
os.makedirs("image", exist_ok=True)

# ===================================================================================
# START: 新增用于统计的函数
# ===================================================================================

def initialize_tool_stats(tools):
    """初始化一个字典，用于存储每个工具的统计数据。"""
    stats = {}
    for tool in tools:
        stats[tool.name] = {"total_calls": 0, "successful_calls": 0}
    return stats

def update_stats_from_run(stats, intermediate_steps):
    """
    解析智能体单次运行的中间步骤，以更新工具使用统计。
    如果工具的返回结果（observation）中包含 'error' 或 'invalid tool' 等关键词，则视为调用失败。
    """
    if not intermediate_steps:
        return

    for step in intermediate_steps:
        
        action,  observation = step
        # 确保这是一个有效的工具动作
        if isinstance(action, AgentAction):
            tool_name = action.tool
            if tool_name in stats:
                stats[tool_name]["total_calls"] += 1
                # 检查返回结果中常见的失败标志
                observation_lower = str(observation).lower()
                if "error" not in observation_lower and "invalid tool" not in observation_lower and "not a valid tool" not in observation_lower:
                    stats[tool_name]["successful_calls"] += 1
            else:
                print(f"[统计] 警告: 工具 '{tool_name}' 被调用，但不在初始工具列表中。")

def write_stats_to_file(stats, file_path):
    
    total_calls_overall = 0
    successful_calls_overall = 0

    with open(file_path, "w", encoding='utf-8') as f:
        f.write("VIoTGPT 工具调用成功率报告\n")
        f.write("=" * 40 + "\n\n")
        
        header = f"{'工具名称':<30} | {'总调用次数':>12} | {'成功调用次数':>15} | {'成功率 (%)':>18}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for tool_name, tool_data in sorted(stats.items()):
            total = tool_data["total_calls"]
            success = tool_data["successful_calls"]
            rate = (success / total * 100) if total > 0 else 0
            
            total_calls_overall += total
            successful_calls_overall += success
            
            line = f"{tool_name:<30} | {total:>12} | {success:>15} | {rate:>18.2f}\n"
            f.write(line)

        f.write("-" * len(header) + "\n")
        
        overall_rate = (successful_calls_overall / total_calls_overall * 100) if total_calls_overall > 0 else 0
        summary_line = f"{'总计':<30} | {total_calls_overall:>12} | {successful_calls_overall:>15} | {overall_rate:>18.2f}\n"
        f.write(summary_line)
        f.write("\n" + "=" * 40 + "\n")
        
    print(f"\n[统计] 工具使用统计报告已成功写入到 {file_path}")



def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split("\n")
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(" "))
        paragraphs = paragraphs[1:]
    return "\n" + "\n".join(paragraphs)


class ConversationBot:
    def __init__(self, load_dict):
        
        print(f"Initializing VIoTGPT_nodemo, load_dict={load_dict}")
        self.models = {}
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k != "self"}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](**{name: self.models[name] for name in template_required_names})
        print(f"All the Available Functions: {self.models}")
        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        if "Qwen2" in args.model_path:
            self.llm = Qwen25VLModel(model_path=args.model_path, lora_adapter_path=args.lora_path, load_in_8bit=True)
        elif "Qwen3" in args.model_path:
            self.llm = Qwen3VLModel(model_path=args.model_path, lora_adapter_path=args.lora_path, load_in_8bit=True)
        elif "LLAMA3" in args.model_path:
            self.llm = Llama3Model(model_path=args.model_path, lora_adapter_path=args.lora_path, load_in_8bit=True)
        
        elif args.lora_path == "":
            self.llm = LlamaModel(args.model_path)
        else:
            self.llm = LoraModel(base_name_or_path=args.model_path, model_name_or_path=args.lora_path, #load_8bit=False)
                                 load_8bit=True)
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key="output")
        self.agent = initialize_agent(self.tools, self.llm, agent="conversational-react-description", verbose=True, memory=self.memory, return_intermediate_steps=True, agent_kwargs={"prefix": PREFIX, "format_instructions": FORMAT_INSTRUCTIONS, "suffix": SUFFIX,}, handle_parsing_errors=True,max_iterations=3,  
            early_stopping_method="generate")

    def parse_agent_log(self, log):
        """
        解析代理的日志，提取 Thought、Action 和 Action Input。
        假设日志遵循 FORMAT_INSTRUCTIONS 定义的格式。
        返回包含这些字段的字典。
        """
        analysis = ""
        thought = ""
        action = ""
        action_input = ""
        observation = ""

        # 基于典型 LangChain 代理日志格式的正则表达式
        analysis_pattern = r"Analysis: (.*?)(?:\n|$)"
        thought_pattern = r"Thought: (.*?)(?:\n|$)"
        action_pattern = r"Action: (.*?)(?:\n|$)"
        action_input_pattern = r"Action Input: (.*?)(?:\n|$)"
        observation_pattern = r"Observation: (.*?)(?:\n|$)"

        
        analysis_match = re.search(analysis_pattern, log, re.DOTALL)
        thought_match = re.search(thought_pattern, log, re.DOTALL)
        action_match = re.search(action_pattern, log, re.DOTALL)
        action_input_match = re.search(action_input_pattern, log, re.DOTALL)
        observation_match = re.search(observation_pattern, log, re.DOTALL)
        
        
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip()
        if action_input_match:
            action_input = action_input_match.group(1).strip()
        if observation_match:
            observation = observation_match.group(1).strip()

        return {
            "Analysis": analysis, 
            "Thought": thought,
            "Action": action,
            "Action Input": action_input,
            "Observation": observation,
        }

    def run_imagetext(self, text):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=1000
        )
        res = self.agent({"input": text.strip()})
        
        # 初始化列表以存储解析后的中间步骤
        inter_logs = []
        parsed_steps = []
        intermediate_steps = res["intermediate_steps"]
        # 处理每个中间步骤
        for step in res["intermediate_steps"]:
            agent_action, tool_output = step
            log = agent_action.log
            inter_logs.append(log)  # 保留原始日志以保持兼容性

            # 解析日志以提取 Thought、Action 和 Action Input
            parsed_log = self.parse_agent_log(log)
            parsed_log["Observation"] = str(tool_output)  # 添加工具输出以完整记录
            parsed_steps.append(parsed_log)

            # 打印解析的信息
            print(f"\n中间步骤：")
            print(f"Thought: {parsed_log['Thought']}")
            print(f"Action: {parsed_log['Action']}")
            print(f"Action Input: {parsed_log['Action Input']}")
            print(f"Observation: {parsed_log['Observation']}")

        # 替换输出中的反斜杠以保持一致性
        res["output"] = res["output"].replace("\\", "/")

        # 如果需要，格式化响应以包含图像 markdown
        response = re.sub(
            "(image/[-\w]*.png)",
            lambda m: f"![](file={m.group(0)})*{m.group(0)}*",
            res["output"],
        )

        print(
            f"\n处理 run_imagetext，输入文本: {text}\n当前内存: {self.agent.memory.buffer}"
        )

        return inter_logs, response, parsed_steps,intermediate_steps
    def run_videotext(self, text):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=1000
        )
        res = self.agent({"input": text.strip()})
        
        # 初始化列表以存储解析后的中间步骤
        inter_logs = []
        parsed_steps = []
        intermediate_steps = res["intermediate_steps"]
        # 处理每个中间步骤
        for step in res["intermediate_steps"]:
            agent_action, tool_output = step
            log = agent_action.log
            inter_logs.append(log)  # 保留原始日志以保持兼容性

            # 解析日志以提取 Thought、Action 和 Action Input
            parsed_log = self.parse_agent_log(log)
            parsed_log["Observation"] = str(tool_output)  # 添加工具输出以完整记录
            parsed_steps.append(parsed_log)

            # 打印解析的信息
            print(f"\n中间步骤：")
            print(f"Thought: {parsed_log['Thought']}")
            print(f"Action: {parsed_log['Action']}")
            print(f"Action Input: {parsed_log['Action Input']}")
            print(f"Observation: {parsed_log['Observation']}")

        # 替换输出中的反斜杠以保持一致性
        res["output"] = res["output"].replace("\\", "/")

        # 如果需要，格式化响应以包含图像 markdown
        response = re.sub(
            "(image/[-\w]*.mp4)",
            lambda m: f"![](file={m.group(0)})*{m.group(0)}*",
            res["output"],
        )

        print(
            f"\n处理 run_videotext，输入文本: {text}\n当前内存: {self.agent.memory.buffer}"
        )

        return inter_logs, response, parsed_steps,intermediate_steps
    def read_intermediate_steps(self, text):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=1000
        )
        res = self.agent({"input": text.strip()})
        
        
        intermediate_steps = res["intermediate_steps"]

        
       
        return  intermediate_steps 

    def run_image(self, image, txt):
        
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        # img = Image.open(image)
        # width, height = img.size
        # ratio = min(512 / width, 512 / height)
        # width_new, height_new = (round(width * ratio), round(height * ratio))
        # width_new = int(np.round(width_new / 64.0)) * 64
        # height_new = int(np.round(height_new / 64.0)) * 64
        # img = img.resize((width_new, height_new))
        # img = img.convert("RGB")
        # img.save(image_filename, "PNG")
        # print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        image_filename = image
        time.sleep(1)

        Human_prompt = (f"\nHuman: provide a figure named {image_filename}. " f"You can use one or several tools to finish following tasks, rather than directly imagine. " f"Especially, you will never use nonexistent tools. " f'Once you have the final answer, do tell me in the format of "Final Answer: [your response here]". \n')
        AI_prompt = f'Received. I will tell you in the format of "Final Answer: [your response here]"'
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The input must only contain the text and the path, without any parameter names. For example, a valid input is a red car,./w_data/street_view.png '
        AI_prompt = f'Received.'
        self.agent.memory.buffer = (self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt)
        print(f"\nProcessed run_image, Input image: {image_filename}\n" f"Current Memory: {self.agent.memory.buffer}")
        return f"{txt} {image_filename} "
    
    def run_video(self, video, txt):
        # video_filename = os.path.join('videos', f"{str(uuid.uuid4())[:8]}.mp4")
        # copyfile(video, video_filename)
        video_filename = video
        Human_prompt = f'\nHuman: provide a video named {video_filename}. ' \
                           f'You can use one or several tools to finish following tasks, rather than directly imagine. ' \
                           f'Especially, you will never use nonexistent tools. ' \
                           f'Once you have the final answer, do tell me in the format of "Final Answer: [your response here]". \n'
        AI_prompt = f'Received. I will tell you in the format of "Final Answer: [your response here]"'
        Human_prompt = f'\nHuman: provide a video named {video_filename}. '
        AI_prompt = f'Received.'
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print(f"\nProcessed video, Input video: {video_filename}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return f'{txt} {video_filename} '
    def run_imageandvideo(self, paths, txt):
       
        try:
            image_path, video_path = [p.strip() for p in paths.split(',')]
        except ValueError:
            print(f"[error] Incorrect run_imageandvideo path: {paths}。It should be 'image_path,video_path'.")
            return txt 

        # --- 处理图像 ---
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        print("======> Image part ...")
        # try:
        #     img = Image.open(image_path)
        #     # (此处省略了图像缩放逻辑，您可以根据需要从 run_image 中复制过来)
        #     width, height = img.size
        #     ratio = min(512 / width, 512 / height)
        #     width_new, height_new = (round(width * ratio), round(height * ratio))
        #     width_new = int(np.round(width_new / 64.0)) * 64
        #     height_new = int(np.round(height_new / 64.0)) * 64
        #     img = img.resize((width_new, height_new))
        #     img = img.convert("RGB")
        #     img.save(image_filename, "PNG")
        #     time.sleep(1)
        #     print(f"图像 {image_path} 已处理并保存为 {image_filename}")
        # except FileNotFoundError:
        #     print(f"[错误] 图像文件未找到: {image_path}")
        #     # 即使图片处理失败，也继续处理视频
        #     image_filename = f"ERROR_FILE_NOT_FOUND_{os.path.basename(image_path)}"
        image_filename = image_path
        # --- 处理视频 ---
        video_filename = os.path.join('videos', f"{str(uuid.uuid4())[:8]}.mp4")
        print("======> Video part ...")
        # try:
        #     copyfile(video_path, video_filename)
        #     print(f"视频 {video_path} 已处理并保存为 {video_filename}")
        # except FileNotFoundError:
        #     print(f"[错误] 视频文件未找到: {video_path}")
        #     # 即使视频处理失败，也要继续
        #     video_filename = f"ERROR_FILE_NOT_FOUND_{os.path.basename(video_path)}"
        video_filename = video_path
        # --- 更新智能体记忆 ---
        # 创建一个组合提示，同时提及图像和视频
        Human_prompt = (
            f'\nHuman: provide a figure named {image_filename} and a video named {video_filename}. '
            f'You can use one or several tools to finish following tasks, rather than directly imagine. '
            f'Especially, you will never use nonexistent tools. '
            f'Once you have the final answer, do tell me in the format of "Final Answer: [your response here]". \n'
        )
        AI_prompt = f'Received. I will tell you in the format of "Final Answer: [your response here]"'
        Human_prompt = f'\nHuman: provide a figure named {image_filename} and a video named {video_filename}. '
        AI_prompt = f'Received.'
        # 将提示添加到记忆中
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        
        print(f"\n已处理图像和视频。输入: {paths}\n"
              f"注册文件: {image_filename}, {video_filename}\n"
              f"当前记忆: {self.agent.memory.buffer}")

        # 返回一个包含两个文件名的文本，以便后续查询使用
        return f'{txt} {image_filename} {video_filename} '

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="FaceDetection_cuda:0")
    parser.add_argument("--model_path", type=str, default="./models/7B_hf/")
    parser.add_argument("--lora_path", type=str, default="", required=False, help="tool-llama lora model path")
    parser.add_argument("--output_path", type=str, default="", required=True, help="Preprocessed tool data output path.")
    parser.add_argument("--query_path", type=str, default="", help="query_path.")
    parser.add_argument("--query_data_path", type=str, default="", help="query_path.")
    
    parser.add_argument(
        "--stats_path",
        type=str,
        default="./evaluation/tool_stats.txt",
        help="Path to save the tool call statistics report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,  # 默认值为 None，表示不限制
        help="只处理数据集中的前 N 个项目。如果不设置，则处理全部项目。",
    )
   
    args = parser.parse_args()
    load_dict = {
        e.split("_")[0].strip(): e.split("_")[1].strip() for e in args.load.split(",")
    }
    print("load_dict", load_dict)
    

    print("[统计] 初始化统计追踪器...")
    # temp_bot = ConversationBot(load_dict=load_dict)
    bot = ConversationBot(load_dict=load_dict)
    tool_stats = initialize_tool_stats(bot.tools)
    # del temp_bot
    torch.cuda.empty_cache()
    gc.collect()
    print("[统计] 追踪器初始化完成。")
    
    
    data_dicts = json.load(open(args.query_path, "r"))
    if args.limit is not None and args.limit > 0:
        print(f"[配置] 已设置处理上限，将只处理 {args.limit} 个项目。")
        data_dicts = data_dicts[:args.limit]
    len_query_file = len(data_dicts)
    out_file = jsonlines.open(args.output_path, "w")
    out_file._flush = True
    
    # ==================== 初始化计时变量 ====================
    total_processing_time = 0.0
    overall_start_time = time.time()
    # =======================================================
    
    for i in range(len_query_file):
        start_time_item = time.time()
        data_dict = data_dicts[i]
        query = data_dict["query"]
        
        raw_img_path = data_dict["img_full_path"] 
        
        
        path_parts = [p.strip() for p in raw_img_path.split(',')]
       
        full_paths = ["{}{}".format(args.query_data_path,p[1:]) if p.startswith('./') else p for p in path_parts]
        img_path = ",".join(full_paths)
       
        print(f"\n[DEBUG] === 正在处理第 {i+1}/{len_query_file} 个项目 ===")
        print(f"[DEBUG] 原始路径字符串: {raw_img_path}")
        print(f"[DEBUG] 处理后的完整路径: {img_path}")
        first_path = img_path.split(',')[0]
        if not os.path.exists(first_path) or os.path.getsize(first_path) == 0:
            print(f"[错误] 第一个源文件不存在或为空: {first_path}")
            continue

        bot.memory.clear()

        intermediate_steps_for_stats = [] 

        
        try:
            print(f"[DEBUG] 正在执行查询: '{query}'")
            
            if "," in img_path:
                
                print("[DEBUG] 检测到图像和视频混合输入。")
                processed_text = bot.run_imageandvideo(img_path, "Processed run_imageandvideo, Input files: ")
              
                inter_logs, response, parsed_steps, intermediate_steps_for_stats = bot.run_imagetext(query) # 或者 run_videotext，取决于你的默认期望
            
            elif img_path.lower().endswith(".mp4"):
                
                print("[DEBUG] 检测到视频输入。")
                processed_text = bot.run_video(img_path, "Processed run_video, Input video: ")
                inter_logs, response, parsed_steps, intermediate_steps_for_stats = bot.run_videotext(query)

            else:
               
                print("[DEBUG] 检测到图像输入。")
                processed_text = bot.run_image(img_path, "Processed run_image, Input image: ")
                inter_logs, response, parsed_steps, intermediate_steps_for_stats = bot.run_imagetext(query)
            
            print("[DEBUG] 处理成功。")
            dict_to_write = {
                "image_name_GT": img_path,
                "id": query,
                "chains": inter_logs,
                "parsed_steps": parsed_steps,
                "result": response,
            }

        except Exception as e:
            print(f"[错误] 处理项目 {i} (图片: {img_path}) 时发生异常")
            print(f"[错误] 异常类型: {type(e)}")
            print(f"[错误] 异常信息: {e}")
            
            dict_to_write = {
                "image_name_GT": img_path,
                "id": query,
                "chains": "ERROR",
                "parsed_steps": [],
                "result": f"An error occurred: {e}",
            }
        finally:
           
            print("[统计] 正在更新本轮的工具调用统计...")
            update_stats_from_run(tool_stats, intermediate_steps_for_stats)
            print(json.dumps(tool_stats, indent=2, ensure_ascii=False))
        
        print("[DEBUG] 正在将结果写入输出文件。")
        print(dict_to_write)
        out_file.write(dict_to_write)
        
        end_time_item = time.time() # 单个条目结束计时
        duration_item = end_time_item - start_time_item
        total_processing_time += duration_item # 累加总时间
        
        # ==================== 打印单个条目耗时 ====================
        print(f"--- [时间统计] 处理第 {i+1}/{len_query_file} 个项目耗时: {duration_item:.2f} 秒 ---")
        # =======================================================
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
    # ==================== 循环结束后，打印总结报告 ====================
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    if len_query_file > 0:
        average_time_per_item = total_processing_time / len_query_file
    else:
        average_time_per_item = 0

    print("\n" + "="*60)
    print("所有项目处理完毕，生成最终报告...")
    print(f"处理项目总数: {len_query_file}")
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"平均每个项目耗时: {average_time_per_item:.2f} 秒")
    print("="*60 + "\n")
        
    
    write_stats_to_file(tool_stats, args.stats_path)
    out_file.close()
    del bot
    torch.cuda.empty_cache()
    gc.collect()