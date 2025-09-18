from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import os
import yaml
from agent_video.base.sampler import AdaptiveSampler
from agent_video.base.preprocess import preprocess_pipeline
from agent_video.routers.errors import ErrorManager
from agent_video.base import get_task_load_status
import time
import requests
import numpy as np
import io

videoRouter = APIRouter()

@videoRouter.post("/video/stream/push")
async def push_stream(request: Request):
    """
    接收HTTP视频流地址，支持多个任务ID和app_names。
    """
    data = await request.json()
    task_ids = data.get("task_ids", [])
    app_names = data.get("app_names", [])
    stream_url = data.get("stream_url")
    camera_id = data.get("camera_id")
    if not stream_url or not task_ids or not app_names:
        return JSONResponse(content=ErrorManager.wrap_data("参数缺失", success=False))
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return JSONResponse(content=ErrorManager.wrap_data("无法打开视频流", success=False))

    # 动态加载所有 app_names 的配置
    configs = {}
    for app_name in app_names:
        yaml_path = os.path.join("config/applications", f"{app_name.replace('_', '-')}.yaml")
        if not os.path.exists(yaml_path):
            continue
        with open(yaml_path, "r", encoding="utf-8") as f:
            configs[app_name] = yaml.safe_load(f)

    # 多任务独立采样与预处理配置
    samplers = {}
    preprocess_steps = {}
    sampling_policies = {}
    batch_sizes = {}
    batch_frames = {}
    for task in task_ids:
        cfg = configs.get(task)
        if not cfg:
            continue
        policy = cfg.get("sampling", {}).get("policy", {})
        default_interval = policy.get("low_load", 1)
        min_interval = 1
        max_interval = policy.get("high_load", 5)
        samplers[task] = AdaptiveSampler(default_interval, min_interval, max_interval)
        # 支持配置文件中 preprocess.steps 列表
        preprocess_cfg = cfg.get("preprocess", {})
        steps = preprocess_cfg.get("steps", [])
        preprocess_steps[task] = steps
        sampling_policies[task] = policy

    frame_count = 0
    sampled_counts = {task: 0 for task in task_ids}

    # 推理API调用地址（Nginx 负载均衡入口，实际部署时替换为真实地址）
    inference_api = "http://nginx_host:8080/infer"

    no_frame_start_time = None

    while True:
        ret, frame = cap.read()
        # 读取失败则等待一段时间，避免死循环
        if not ret:
            now = time.time()
            if no_frame_start_time is None:
                no_frame_start_time = now
            elif now - no_frame_start_time > 300:
                break
            continue
        no_frame_start_time = None
        frame_count += 1
        for task in task_ids:
            # 先根据各任务采样策略判断是否累计帧
            sampler = samplers.get(task)
            if sampler and not sampler.should_sample():
                continue  # 不满足采样率，跳过累计
            
            # 满足采样策略的帧才累计到buffer
            if task not in batch_frames:
                batch_frames[task] = []
            batch_frames[task].append(frame)
            
            # 满足batch_size时才处理
            if len(batch_frames[task]) >= batch_sizes.get(task, 1):
                # 采样状态调节
                policy = sampling_policies.get(task, {})
                load_status = get_task_load_status(task)
                if sampler:
                    sampler.update_interval(load_status, policy)
                # 图像预处理
                steps = preprocess_steps.get(task, [])
                processed_batch = preprocess_pipeline(batch_frames[task], steps)
                # 推理（通过Nginx发送）
                files = []
                for idx, img in enumerate(processed_batch):
                    break
                try:
                    response = requests.post(inference_api)
                except Exception as e:
                    print(f"推理请求失败: {e}")
                batch_frames[task] = []
    cap.release()
    return JSONResponse(content={})

@videoRouter.post("/video/stream/upload")
async def upload_video(request: Request, file: UploadFile = File(...)):
    """
    上传视频文件，支持多个任务ID和app_names。
    参数通过form-data: task_ids, app_names, camera_id
    """
    return
