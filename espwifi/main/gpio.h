#ifndef GPIO_H
#define GPIO_H

#include <stddef.h>
#include <stdint.h>
#include "driver/uart.h"

/* UART 端口号 — 供其他模块引用 */
#define UART_PORT       UART_NUM_2

void uart_init(void);


#endif