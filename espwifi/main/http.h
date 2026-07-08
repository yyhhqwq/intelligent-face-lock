#ifndef HTTP_H
#define HTTP_H

#include "esp_err.h"

/**
 * @brief HTTP 上报服务器地址 — 请修改为你自己的服务器
 */
#define HTTP_POST_URL   "http://43.131.251.209:8080/api/upload"

/**
 * @brief 上报一条 JSON 数据到服务器
 * @param json_data 需要上报的 JSON 字符串
 * @return ESP_OK 成功，否则失败
 */
esp_err_t http_post_json(const char *json_data);

#endif
