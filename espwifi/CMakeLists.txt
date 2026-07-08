cmake_minimum_required(VERSION 3.16)

# 设置项目名称，后续可用 idf.py set-target esp32s3 设置目标芯片
set(EXTRA_COMPONENT_DIRS "")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# 项目名称 — 你可以改成自己想要的名字
project(espwifi)
