#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_http_client.h"
#include "http.h"

static const char *TAG = "HTTP";

/**
 * @brief HTTP 事件回调 — 仅用于日志
 */
static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    switch (evt->event_id) {
    case HTTP_EVENT_ON_CONNECTED:
        ESP_LOGD(TAG, "HTTP 已连接");
        break;
    case HTTP_EVENT_ON_FINISH:
        ESP_LOGD(TAG, "HTTP 请求完成");
        break;
    case HTTP_EVENT_DISCONNECTED:
        ESP_LOGD(TAG, "HTTP 断开连接");
        break;
    case HTTP_EVENT_ERROR:
        ESP_LOGE(TAG, "HTTP 错误");
        break;
    default:
        break;
    }
    return ESP_OK;
}

esp_err_t http_post_json(const char *json_data)
{
    if (json_data == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_http_client_config_t config = {
        .url = HTTP_POST_URL,
        .method = HTTP_METHOD_POST,
        .event_handler = http_event_handler,
        .timeout_ms = 5000,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL) {
        ESP_LOGE(TAG, "HTTP 客户端初始化失败");
        return ESP_FAIL;
    }

    /* 设置请求头 */
    esp_http_client_set_header(client, "Content-Type", "application/json");
    /* 设置请求体 */
    esp_http_client_set_post_field(client, json_data, strlen(json_data));

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int status_code = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "HTTP 上报成功 | 状态码: %d | 数据: %.100s",
                 status_code, json_data);
    } else {
        ESP_LOGE(TAG, "HTTP 上报失败 | err=%s | 数据: %.80s",
                 esp_err_to_name(err), json_data);
    }

    esp_http_client_cleanup(client);
    return err;
}
