#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "gpio.h"
#include "wifi.h"
#include "http.h"

static const char *TAG = "MAIN";

/* UART 接收缓冲区大小 */
#define RX_BUF_SIZE     2048
/* 单行最大长度 */
#define LINE_MAX_LEN    1024


/**
 * @brief 简单的 JSON 格式校验
 *        检查字符串是否以 '{' 开头、'}' 结尾
 * @return true 看起来像 JSON
 */
static bool looks_like_json(const char *str)
{
    if (str == NULL) return false;

    /* 跳过前导空格 */
    while (*str == ' ' || *str == '\t') str++;
    if (*str != '{') return false;

    /* 找到末尾 */
    int len = strlen(str);
    if (len == 0) return false;
    const char *end = str + len - 1;
    /* 跳过尾部空格和换行 */
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) end--;

    return *end == '}';
}


/**
 * @brief UART 监听任务 — 只读取并打印，不转发 HTTP
 *        用于阶段一和阶段二，让用户能看到 UART 有数据进来
 */
static void uart_monitor_task(void *pvParameters)
{
    /* 静态分配，避免栈溢出 */
    static char raw_buf[RX_BUF_SIZE];
    static char line_buf[LINE_MAX_LEN];
    int line_pos = 0;

    while (1) {
        int len = uart_read_bytes(UART_PORT, (uint8_t *)raw_buf, sizeof(raw_buf) - 1,
                                  pdMS_TO_TICKS(100));
        if (len <= 0) continue;
        raw_buf[len] = '\0';

        for (int i = 0; i < len; i++) {
            char ch = raw_buf[i];
            if (ch == '\n') {
                if (line_pos > 0) {
                    line_buf[line_pos] = '\0';
                    ESP_LOGI(TAG, "[UART] %s", line_buf);
                    line_pos = 0;
                }
            } else if (ch != '\r') {
                if (line_pos < LINE_MAX_LEN - 1)
                    line_buf[line_pos++] = ch;
                else
                    line_pos = 0;
            }
        }
    }
}


/**
 * @brief UART 接收 + JSON 解析 + HTTP 上报任务
 *
 * 从 UART2 读取数据，按行分割，提取完整 JSON 后通过 HTTP POST 上报
 */
static void uart_forward_task(void *pvParameters)
{
    /* 静态分配，避免栈溢出 */
    static char raw_buf[RX_BUF_SIZE];
    static char line_buf[LINE_MAX_LEN];
    int line_pos = 0;

    while (1) {
        /* 从 UART2 读取原始数据 */
        int len = uart_read_bytes(UART_PORT, (uint8_t *)raw_buf, sizeof(raw_buf) - 1,
                                  pdMS_TO_TICKS(100));
        if (len <= 0) {
            continue;
        }
        raw_buf[len] = '\0';

        /* 逐字节处理，按行分割 */
        for (int i = 0; i < len; i++) {
            char ch = raw_buf[i];

            if (ch == '\n') {
                /* 遇到换行符，处理积累的一行 */
                if (line_pos > 0) {
                    line_buf[line_pos] = '\0';

                    /* 检查是否为 JSON 格式 */
                    if (looks_like_json(line_buf)) {
                        ESP_LOGI(TAG, "📤 上报 [%d bytes]: %.120s", line_pos, line_buf);
                        http_post_json(line_buf);
                    } else if (line_buf[0] != '\0') {
                        ESP_LOGW(TAG, "⏭️ 非 JSON 数据，跳过: %s", line_buf);
                    }
                    line_pos = 0;
                }
            } else if (ch != '\r') {
                /* 忽略回车，其他字符存入行缓冲区 */
                if (line_pos < LINE_MAX_LEN - 1) {
                    line_buf[line_pos++] = ch;
                } else {
                    /* 行太长，丢弃 */
                    ESP_LOGW(TAG, "行超长，丢弃");
                    line_pos = 0;
                }
            }
        }

        /* 如果长时间没有换行且缓冲区有内容，尝试处理（兼容无换行数据） */
        if (line_pos > 200) {
            line_buf[line_pos] = '\0';
            if (looks_like_json(line_buf)) {
                ESP_LOGI(TAG, "📤 上报(超时) [%d bytes]: %.120s", line_pos, line_buf);
                http_post_json(line_buf);
            }
            line_pos = 0;
        }
    }
}

/* ===================================================
 *  逐步验证配置
 * ===================================================
 * 阶段一: 只验证 WiFi 连接 (将下面设为 1)
 * 阶段二: 验证 WiFi + HTTP 发送测试数据 (设为 2)
 * 阶段三: 全流程 UART → HTTP (设为 3)
 */
#define VERIFICATION_STAGE  3

/* 测试用的样本 JSON */
#define TEST_JSON_STR   "{\"type\":\"test\",\"msg\":\"ESP32-S3 HTTP test OK\",\"time\":\"2026-06-25 12:00:00\"}"


#if VERIFICATION_STAGE >= 1
/**
 * @brief 阶段一: 验证 WiFi 连接 + UART 监听
 */
static void verify_wifi(void)
{
    ESP_LOGI(TAG, "═══════ 阶段一: WiFi 连接验证 ═══════");

    /* 启动 UART 监听（只打印，不转发） */
    xTaskCreate(uart_monitor_task, "uart_mon", 8192, NULL, 10, NULL);
    ESP_LOGI(TAG, "UART 监听已启动，等待 K230 数据...");

    esp_err_t ret = wifi_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "✅ [PASS] WiFi 连接成功!");
        ESP_LOGI(TAG, "   SSID: %s", WIFI_SSID);
    } else {
        ESP_LOGE(TAG, "❌ [FAIL] WiFi 连接失败!");
        ESP_LOGE(TAG, "   请检查 wifi.h 中的 SSID 和密码是否正确");
    }
}
#endif

#if VERIFICATION_STAGE >= 2
/**
 * @brief 阶段二: 验证 HTTP POST + UART 监听
 */
static void verify_http(void)
{
    ESP_LOGI(TAG, "═══════ 阶段二: HTTP 上报验证 ═══════");

    /* UART 监听在阶段一已启动，这里直接测试 HTTP */

    if (!wifi_is_connected()) {
        ESP_LOGE(TAG, "❌ [SKIP] WiFi 未连接，跳过 HTTP 测试");
        return;
    }

    ESP_LOGI(TAG, "📤 发送测试数据到: %s", HTTP_POST_URL);
    ESP_LOGI(TAG, "   数据: %s", TEST_JSON_STR);

    esp_err_t ret = http_post_json(TEST_JSON_STR);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "✅ [PASS] HTTP 上报成功! 请检查服务器是否收到数据");
    } else {
        ESP_LOGE(TAG, "❌ [FAIL] HTTP 上报失败!");
        ESP_LOGE(TAG, "   请检查:");
        ESP_LOGE(TAG, "   1. 服务器地址 %s 是否正确", HTTP_POST_URL);
        ESP_LOGE(TAG, "   2. 服务器是否已启动并监听该端口");
        ESP_LOGE(TAG, "   3. ESP32-S3 和服务器是否在同一网络");
    }
}
#endif

#if VERIFICATION_STAGE >= 3
/**
 * @brief 阶段三: 开始完整的 UART → HTTP 转发
 */
static void verify_full_forward(void)
{
    ESP_LOGI(TAG, "═══════ 阶段三: UART → HTTP 全流程 ═══════");
    ESP_LOGI(TAG, "等待 K230 通过 UART2 发送数据...");
    ESP_LOGI(TAG, "收到 JSON 后将自动转发到 %s", HTTP_POST_URL);

    /* 切换到 HTTP 转发任务（替代监听任务） */
    xTaskCreate(uart_forward_task, "uart_fwd", 8192, NULL, 10, NULL);
}
#endif


void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   ESP32-S3 UART → HTTP 转发器 v1.0   ║");
    ESP_LOGI(TAG, "║   验证阶段: %d/3                      ║", VERIFICATION_STAGE);
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* 0. 初始化 UART2 */
    uart_init();

    /* ===== 逐步验证 ===== */
#if VERIFICATION_STAGE >= 1
    verify_wifi();
#endif

#if VERIFICATION_STAGE >= 2
    vTaskDelay(pdMS_TO_TICKS(500));
    verify_http();
#endif

#if VERIFICATION_STAGE >= 3
    vTaskDelay(pdMS_TO_TICKS(500));
    verify_full_forward();
#endif

    /* 主循环 — 空闲，打印系统状态 */
    while (1) {
        ESP_LOGI(TAG, "🟢 系统运行中 | WiFi: %s | 等待 UART 数据...",
                 wifi_is_connected() ? "已连接" : "未连接");
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}