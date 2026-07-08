"""
GPIOA0 驱动模块 & 串口通信模块

提供 GPIOA0 引脚的低电平触发控制。
提供 UART 串口通信功能（GPIO11=TX, GPIO12=RX）。
引脚编号可通过参数动态传入。
"""

from machine import Pin, UART, FPIOA


class GPIOA0Controller:
    """GPIOA0 控制器，支持低电平触发。"""

    def __init__(self, pin=37):
        self.pin = None
        self.pin_num = pin
        try:
            self.pin = Pin(pin, Pin.OUT)
            self.pin.value(1)  # 默认高电平（未触发）
            print("[GPIOA0] 初始化成功，引脚: {}".format(pin))
        except Exception as e:
            print("[错误] GPIOA0 Pin({}) 初始化失败: {}".format(pin, e))

    def trigger(self):
        """输出低电平触发"""
        if self.pin is None:
            print("[错误] GPIOA0 未初始化")
            return False
        try:
            self.pin.value(0)
            return True
        except Exception as e:
            print("[错误] GPIOA0 触发失败: {}".format(e))
            return False

    def release(self):
        """恢复高电平（释放）"""
        if self.pin is None:
            return False
        try:
            self.pin.value(1)
            return True
        except Exception:
            return False

    def get_state(self):
        """读取当前引脚电平状态"""
        if self.pin is None:
            return None
        try:
            return self.pin.value()  # 返回 0 或 1
        except Exception:
            return None

    def deinit(self):
        """释放引脚资源"""
        if self.pin is not None:
            try:
                self.pin.value(1)
            except Exception:
                pass
            self.pin = None


def trigger_gpioa0(pin=37):
    """快捷方法：驱动 GPIOA0 输出低电平触发
    
    Args:
        pin: 引脚编号，默认 37
    
    Returns:
        GPIOA0Controller 实例
    """
    ctrl = GPIOA0Controller(pin)
    ctrl.trigger()
    return ctrl


class UARTLogger:
    """串口日志发送器，通过 UART 将日志信息发送到外部设备。
    
    引脚映射（根据硬件图）：
        GPIO11 = UART2_TX（发送）
        GPIO12 = UART2_RX（接收）
    """
    
    def __init__(self, baudrate=115200):
        self.uart = None
        self.baudrate = baudrate
        try:
            # 手动配置 FPIOA 映射（K230 CanMV 需要手动切换引脚功能）
            fpioa = FPIOA()
            fpioa.set_function(11, FPIOA.UART2_TXD)
            fpioa.set_function(12, FPIOA.UART2_RXD)
            
            # 初始化 UART（不带 tx/rx 参数）
            self.uart = UART(2, baudrate=baudrate)
            print("[UART] 初始化成功，UART2 GPIO11(TX)/GPIO12(RX) 波特率:{}".format(baudrate))
        except Exception as e:
            print("[错误] UART 初始化失败: {}".format(e))
    
    def send(self, data):
        """发送数据到串口
        
        Args:
            data: 字符串或字节数据
        """
        if self.uart is None:
            return False
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            self.uart.write(data)
            return True
        except Exception as e:
            print("[错误] UART 发送失败: {}".format(e))
            return False
    
    def send_line(self, msg):
        """发送一行日志（自动添加换行符）
        
        Args:
            msg: 日志消息字符串
        """
        return self.send(msg + "\n")
    
    def send_json(self, data_dict):
        """发送 JSON 格式数据
        
        Args:
            data_dict: 字典数据
        """
        try:
            import ujson
            json_str = ujson.dumps(data_dict)
            return self.send_line(json_str)
        except Exception as e:
            print("[错误] UART JSON 发送失败: {}".format(e))
            return False
    
    def read(self, count=1):
        """从串口读取数据
        
        Args:
            count: 读取字节数
        """
        if self.uart is None:
            return None
        try:
            return self.uart.read(count)
        except Exception:
            return None
    
    def any(self):
        """检查是否有可读数据"""
        if self.uart is None:
            return 0
        try:
            return self.uart.any()
        except Exception:
            return 0
    
    def deinit(self):
        """释放串口资源"""
        if self.uart is not None:
            try:
                self.uart.deinit()
            except Exception:
                pass
            self.uart = None
