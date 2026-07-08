# Code 项目仓库

K230 + ESP32-S3 智能人脸识别门禁/考勤系统的完整工程集合，包含设备端 AI 识别、串口数据网关和云端数据接收服务器三部分。

## 项目列表

| 项目 | 平台 | 说明 |
|------|------|------|
| [random](random/) | K230 CanMV (MicroPython) | 人脸识别门禁/考勤系统主程序 |
| [espwifi](espwifi/) | ESP32-S3 (ESP-IDF) | UART 数据 WiFi 网关，转发串口数据到服务器 |
| [httpsever](httpsever/) | Python (FastAPI) | 数据接收服务器，提供 Web 管理界面 |

## 系统架构

```
┌─────────────────┐     UART      ┌─────────────────┐     HTTP      ┌─────────────────┐
│                 │  (GPIO11/12)  │                 │  (WiFi POST)  │                 │
│   K230 CanMV    │ ────────────► │    ESP32-S3     │ ────────────► │   FastAPI 服务器 │
│  人脸识别门禁    │               │   串口数据网关   │               │  数据存储/Web展示 │
│                 │               │                 │               │                 │
└─────────────────┘               └─────────────────┘               └─────────────────┘
```

### 数据流向

1. K230 运行人脸识别，产生识别/通行/报警事件
2. K230 通过 UART 串口发送 JSON 格式事件数据
3. ESP32-S3 接收串口数据，通过 WiFi 以 HTTP POST 上报到服务器
4. 服务器接收并存储数据，提供 Web 界面查询管理

---

## random — K230 人脸识别门禁系统

基于 CanMV K230 平台的智能人脸识别门禁/考勤系统（MicroPython），具备人脸检测、识别、注册、删除、活体检测、门禁控制、考勤日志、IoT 上报等完整功能。

### 核心功能

| 模块 | 说明 |
|------|------|
| 人脸检测 | KPU 硬件加速，320x320 输入，置信度 ≥ 0.5 |
| 人脸识别 | 112x112 输入，相似度 ≥ 0.75 判定已知人脸 |
| 人脸注册 | 按键 A 触发，矩阵键盘输入编号，采集 30 帧特征，上限 50 人 |
| 人脸删除 | 按键 B 触发，安全过滤防路径遍历 |
| 活体检测 | 106 点关键点，支持眨眼/点头/张嘴，随机动作挑战 |
| 门禁控制 | 继电器（GPIO63）、蜂鸣器（GPIO62）、LED（GPIO61） |
| 考勤日志 | SD 卡 CSV 存储，IoT 定时上传 |
| 事件控制 | 异步处理开门/蜂鸣/日志，不阻塞 AI 主循环 |

### 关键文件

| 文件 | 说明 |
|------|------|
| [main.py](random/main.py) | 主程序入口，统一调度 Camera/KPU/Display |
| [recognition.py](random/recognition.py) | 人脸识别模块（检测+识别+关键点） |
| [registration.py](random/registration.py) | 人脸注册模块 |
| [deletion.py](random/deletion.py) | 人脸删除模块 |
| [liveness.py](random/liveness.py) | 活体检测模块（多动作） |
| [hardware.py](random/hardware.py) | 硬件控制（门锁/蜂鸣器/LED） |
| [event_control.py](random/event_control.py) | 事件控制（异步开门/日志） |
| [iot.py](random/iot.py) | IoT 日志上传模块 |
| [kpu_det.py](random/kpu_det.py) | KPU 人脸检测封装 |
| [gpio.py](random/gpio.py) | GPIO/UART 通信模块 |
| [rtc_set.py](random/rtc_set.py) | RTC 时间校准 |
| [send_log.py](random/send_log.py) | 日志发送测试工具 |

### 测试脚本

| 文件 | 说明 |
|------|------|
| [test_uart.py](random/test_uart.py) | UART 通信测试 |
| [test_uart_loopback.py](random/test_uart_loopback.py) | UART 回环测试 |
| [test_uart_pins.py](random/test_uart_pins.py) | UART 引脚测试 |
| [test_sw_uart.py](random/test_sw_uart.py) | 软件串口测试 |
| [test_fpioa.py](random/test_fpioa.py) | FPIOA 引脚映射测试 |
| [test_connectivity.py](random/test_connectivity.py) | 连通性测试 |
| [uart_test_random.py](random/uart_test_random.py) | UART 随机数据测试 |

### 技术参数

- **检测模型**：`face_detection_320.kmodel`（320x320）
- **识别模型**：`face_recognition.kmodel`（112x112）
- **关键点模型**：`face_landmark.kmodel`（192x192，106 点）
- **人脸库容量**：上限 50 人
- **NMS 阈值**：0.2
- **活体超时**：6 秒

---

## espwifi — ESP32-S3 WiFi 网关

基于 ESP-IDF 的 ESP32-S3 串口数据 WiFi 网关，接收 K230 的 UART 数据并通过 HTTP POST 上报到服务器。

### 核心功能

- UART2 串口数据接收（TX=GPIO17, RX=GPIO16, 115200bps）
- JSON 格式自动识别与过滤
- WiFi STA 模式连接，支持断线重连（最多 5 次）
- HTTP POST 数据上报到服务器
- 分阶段验证模式（WiFi → HTTP → 全流程）

### 关键文件

| 文件 | 说明 |
|------|------|
| [main.c](espwifi/main/main.c) | 主程序，UART 接收 + JSON 解析 + HTTP 上报 |
| [wifi.c](espwifi/main/wifi.c) / [wifi.h](espwifi/main/wifi.h) | WiFi 连接管理（SSID/密码配置） |
| [http.c](espwifi/main/http.c) / [http.h](espwifi/main/http.h) | HTTP POST 上报（服务器地址配置） |
| [gpio.c](espwifi/main/gpio.c) / [gpio.h](espwifi/main/gpio.h) | UART 驱动（引脚/波特率配置） |

### 引脚配置

| 功能 | GPIO |
|------|------|
| UART2 TX | GPIO17 |
| UART2 RX | GPIO16 |

### 使用方法

```bash
# 修改 wifi.h 中的 SSID 和密码
# 修改 http.h 中的服务器地址

idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```

详细文档：[espwifi/README.md](espwifi/README.md)

---

## httpsever — 数据接收服务器

基于 FastAPI 的数据接收服务器，接收设备上报的事件数据，提供 Web 管理界面查询和管理。

### 核心功能

- 接收设备 HTTP POST 上报的 JSON 数据
- SQLite 数据库存储设备信息和事件日志
- Web 界面展示统计数据（今日通行、在线设备、活体通过率）
- 设备管理（注册、状态跟踪）
- 事件查询与统计
- CORS 跨域支持

### 关键文件

| 文件 | 说明 |
|------|------|
| [server.py](httpsever/server.py) | FastAPI 服务器主程序 |
| [database.py](httpsever/database.py) | SQLite 数据库操作模块 |
| [requirements.txt](httpsever/requirements.txt) | Python 依赖 |
| [static/index.html](httpsever/static/index.html) | Web 管理界面 |
| [static/js/app.js](httpsever/static/js/app.js) | 前端交互逻辑 |
| [static/css/style.css](httpsever/static/css/style.css) | 样式表 |

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 管理界面 |
| GET | `/health` | 健康检查 |
| POST/PUT/PATCH | `/api/upload` | 接收设备上报数据 |

### 使用方法

```bash
cd httpsever
pip install -r requirements.txt
python server.py
# 访问 http://localhost:8080
```

### 数据库结构

- **devices** 表：设备信息（device_id, name, location, last_seen, status）
- **events** 表：事件日志（device_id, event_type, event_category, name, score, message, details, device_time, created_at, status）

---

## 技术栈总览

| 层级 | 技术 |
|------|------|
| AI 设备端 | K230 CanMV, MicroPython, KPU, nncase |
| 网关 | ESP32-S3, ESP-IDF, FreeRTOS |
| 服务器 | Python, FastAPI, SQLite, Uvicorn |
| 通信协议 | UART, HTTP/POST, WiFi |
| 数据格式 | JSON |

## 目录结构

```
code/
├── random/              # K230 人脸识别门禁系统
│   ├── main.py          # 主程序
│   ├── recognition.py   # 人脸识别
│   ├── registration.py  # 人脸注册
│   ├── deletion.py      # 人脸删除
│   ├── liveness.py      # 活体检测
│   ├── hardware.py      # 硬件控制
│   ├── event_control.py # 事件控制
│   ├── iot.py           # IoT 上传
│   ├── kpu_det.py       # KPU 检测
│   ├── gpio.py          # GPIO/UART
│   ├── rtc_set.py       # RTC 校准
│   ├── send_log.py      # 日志测试工具
│   └── test_*.py        # 测试脚本
├── espwifi/             # ESP32-S3 WiFi 网关
│   ├── main/            # 源代码
│   ├── CMakeLists.txt   # 构建配置
│   └── README.md        # 详细文档
├── httpsever/           # 数据接收服务器
│   ├── server.py        # FastAPI 服务器
│   ├── database.py      # 数据库模块
│   ├── static/          # Web 前端
│   └── requirements.txt # Python 依赖
└── README.md            # 本文件
```

