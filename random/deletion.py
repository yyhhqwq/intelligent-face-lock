"""
===============================================================================
人脸库删除/查询工具
===============================================================================
"""

from libs.PipeLine import ScopedTiming
import os


def _join_path(directory, filename):
    if not directory:
        directory = "/"
    if directory.endswith("/"):
        return directory + filename
    return directory + "/" + filename


def _safe_name(name):
    if name is None:
        return ""
    name = str(name).strip()
    # 只允许竞赛编号常用字符，防止传入 '../xxx' 造成误删。
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    out = ""
    for ch in name:
        if ch in allowed:
            out += ch
    return out


def delete_face_by_name(database_dir, person_name):
    with ScopedTiming("delete_face", True):
        safe_name = _safe_name(person_name)
        if not safe_name:
            print("[删除] 编号为空或包含非法字符，拒绝删除")
            return False

        db_file_path = _join_path(database_dir, safe_name + ".bin")
        try:
            os.remove(db_file_path)
            print("[删除] 已删除人脸: {}".format(safe_name))
            return True
        except Exception as e:
            print("[删除] 人脸 {} 不存在或删除失败: {}".format(safe_name, e))
            return False


def list_registered_faces(database_dir):
    with ScopedTiming("list_faces", True):
        try:
            db_file_list = os.listdir(database_dir)
            face_names = []
            for db_file in db_file_list:
                try:
                    if db_file.endswith(".bin"):
                        face_names.append(db_file[:-4])
                except Exception:
                    pass

            if len(face_names) == 0:
                print("[人脸库] 数据库为空，无已注册人脸")
                return []

            print("[人脸库] 已注册人脸 ({}人):".format(len(face_names)))
            for name in face_names:
                print("  - {}".format(name))
            return face_names
        except Exception as e:
            print("[人脸库] 读取数据库失败: {}".format(e))
            return []


def reset_database(database_dir):
    with ScopedTiming("database_reset", True):
        try:
            print("[人脸库] 正在清空数据库...")
            db_file_list = os.listdir(database_dir)
            count = 0
            for db_file in db_file_list:
                try:
                    if db_file.endswith(".bin"):
                        os.remove(_join_path(database_dir, db_file))
                        count += 1
                except Exception as e:
                    print("[人脸库] 删除 {} 失败: {}".format(db_file, e))
            print("[人脸库] 数据库清空完成，共删除 {} 个人脸".format(count))
            return True
        except Exception as e:
            print("[人脸库] 清空数据库失败: {}".format(e))
            return False
