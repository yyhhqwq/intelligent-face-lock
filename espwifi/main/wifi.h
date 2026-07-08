#ifndef WIFI_H
#define WIFI_H

#include "esp_err.h"

/**
 * @brief WiFi 配置 - 请修改为你自己的热点信息
 */
#define WIFI_SSID       "TP-LINK_FC20"
#define WIFI_PASS       "admin603"
#define WIFI_MAX_RETRY  5

/**
 * @brief 初始化 WiFi 并等待连接成功
 * @return ESP_OK 连接成功，否则失败
 */
esp_err_t wifi_init(void);

/**
 * @brief 获取 WiFi 连接状态
 * @return true 已连接，false 未连接
 */
bool wifi_is_connected(void);

#endif
