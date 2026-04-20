#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import textwrap
# 修改导入方式
from composer import Composer


def test_composer():
    """测试作曲家类的功能"""

    # 初始化作曲家类
    config_path = "config/gpt_config.json"
    composer = Composer(config_path)

    # 测试数据
    lyrics = textwrap.dedent(
        """
        远方的朋友一路辛苦
        请你喝一杯下马酒
        洗去一路风尘
        来看看美丽的草原
        远方的朋友
        尊贵的客人
        献上洁白的哈达
        献上一片草原的深情
        请你喝一杯下马酒
        远方的朋友一路辛苦
        请你喝一杯下马酒
        草原就是你的家
        来尝尝香甜的美酒
        远方的朋友
        尊贵的客人
        献上洁白的哈达
        献上一片草原的深情
        请你喝一杯下马酒
        献上一片草原的深情
        请你喝一杯下马酒啊
    """
    ).strip()

    song_description = "slow, tenor, cello, male, relaxing, traditional Chinese folk, 50s"
    print("=== 测试 LRC 生成功能 ===")
    try:
        lrc_result = composer.generate_lrc(lyrics, song_description)
        print("生成的 LRC 内容:")
        print(lrc_result)
        print()

        print("=== 测试歌曲总时长获取功能 ===")
        duration = composer.get_song_duration(lrc_result)
        print(f"歌曲总时长: {duration} 秒")
        print()

        print("=== 测试去除结构化标签功能 ===")
        lrc_without_tags = composer.remove_structural_tags(lrc_result)
        print("去除结构化标签后的 LRC 内容:")
        print(lrc_without_tags)

    except Exception as e:
        print(f"测试过程中出现错误: {e}")


if __name__ == "__main__":
    test_composer()
