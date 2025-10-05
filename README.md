# 基于海康 VisionMaster 视觉平台的高精度工业视觉智能识别引擎

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![C#](https://img.shields.io/badge/C%23-.NET%206.0-purple?logo=csharp&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi&logoColor=white)
![VisionMaster](https://img.shields.io/badge/VisionMaster-SDK-orange)
![DeepSeek](https://img.shields.io/badge/DeepSeek-AI%20Model-yellow?logo=googlecloud&logoColor=white)

## 🧠 项目简介
本项目旨在构建一个 **高精度工业视觉智能识别引擎**，依托 **海康威视 VisionMaster 平台**，实现工业生产线的自动化检测、分类与识别任务。  
系统采用 **Python + C# 混合架构**，结合 **FastAPI 后端框架** 与 **海康 SDK**，并接入 **DeepSeek 大模型** 实现图像语义理解，为工业场景提供智能化、可扩展的视觉解决方案。

---

## ⚙️ 系统架构

### 🔹 Python 模块（智能识别引擎）
- 使用 **FastAPI** 构建后端服务；
- 对接 **DeepSeek 多模态视觉大模型**；
- 负责图像语义理解、目标检测、分类结果生成；
- 通过 RESTful API 提供接口服务，供 C# 模块调用。

### 🔹 C# 模块（工业流程与控制）
- 基于 **海康 VisionMaster SDK**；
- 实现工业视觉流程的加载、执行与检测任务；
- 控制相机采集、图像预处理、触发检测；
- 与 Python 后端通信，获取识别结果并进行显示与记录。

---

## 🧩 技术栈

| 模块 | 使用语言 / 框架 | 主要功能 |
|------|----------------|-----------|
| 智能识别 | Python 3.x + FastAPI | 对接 DeepSeek，负责图像语义识别 |
| 视觉控制 | C# + VisionMaster SDK | 实现工业流程与检测自动化 |
| 通信机制 | HTTP / JSON | 实现前后端数据交互 |
| 图像分析 | DeepSeek Vision | 实现目标检测与分类理解 |
| 数据输出 | JPG / Log | 输出检测结果与日志信息 |

---

## 📂 项目结构
project_root/
├── backend/ # Python (FastAPI) 模块
│ ├── main.py
│ ├── model_handler.py
│ └── requirements.txt
├── vision_control/ # C# 模块（VisionMaster 控制）
│ ├── VisionControl.cs
│ └── App.config
├── results/
│ └── Result.jpg # 分类检测结果
├── README.md
└── ...

---

## 🚀 使用说明

### 1️⃣ 启动 Python 后端服务
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

服务启动后将运行在 http://127.0.0.1:8000

可通过 /predict 接口接收来自 C# 的图像数据请求，并返回识别结果。

### 2️⃣ 启动 C# 工程

在 Visual Studio 中打开 VisionControl.csproj；

确保已安装 海康 VisionMaster SDK；

运行程序后系统将：

自动加载工业检测流程；

采集图像并调用 FastAPI 接口；

展示分类检测结果与识别日志。


📞 团队信息

团队成员：DobEtooS，JURUO TXT，zzh
项目名称：基于海康 VisionMaster 视觉平台的高精度工业视觉智能识别引擎
开发语言：Python、C#
后端框架：FastAPI
视觉平台：海康威视 VisionMaster
智能模型：DeepSeek Vision

