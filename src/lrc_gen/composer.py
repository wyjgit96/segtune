import json
import re
from typing import Optional
from openai import AzureOpenAI

system_prompt = """
You are a professional music composer and vocal arranger.

Your task:
1. Analyze the lyrics and the song description below.
2. For each line of lyrics, estimate a reasonable singing duration. Base your estimation jointly on:
   - The intrinsic characteristics of the line itself (e.g., length, phrasing, complexity)
   - The overall song attributes;
   - The structural flow of the song, including instrumental breaks, natural pauses, and transitions
3. Identify and insert appropriate structural labels that define the organization of the song, such as:  
   [Start], [End], [Intro], [Outro], [Break], [Bridge], [Inst], [Solo], [Verse], [Chorus]
   - Each structural label should appear on its own line
   - Labels should be assigned timestamps and durations just like lyrics
   - Their placement and duration must be musically reasonable and coherent with the song flow
   - The first line is always the tag [Start] and the last line is always [End].
4. Return: Output a complete `.lrc` style list with timestamps, starting at 00:00.00

Return example:

```lrc
[00:00.00][Start]
[00:01.00][Intro]
[00:07.00]让我掉下眼泪的
...
[03:30.00][End]
```

Rules:
- All timestamps must be in mm:ss.xx format (centisecond precision)
- Each timestamp = previous timestamp + previous line's estimated duration
- Structural labels must match the set: [Start], [End], [Intro], [Outro], [Break], [Bridge], [Inst], [Solo], [Verse], [Chorus]
- Each label must occupy its own line
- Lyrics in the lrc must match the original input
- Do not return explanations or reasoning, only the `.lrc` output
"""

user_template = """
Output the lrc format file with timestamps.

### Song Description
{song_description}

### Lyrics
{lyrics}
"""


class Composer:
    """作曲家类，用于生成 LRC 格式的歌词时间轴和处理结构化标签"""

    def __init__(self, config_path: str):
        """初始化作曲家类
        
        Args:
            config_path: Azure OpenAI 配置文件路径
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_API_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]
        )
        self.deployment_name = config["AZURE_OPENAI_DEPLOYMENT_NAME"]

    def build_lrc_prompt(self, lyrics: str, song_description: str) -> tuple[str, str]:
        """构建用于生成 LRC 的提示
        
        Args:
            lyrics: 歌词内容
            song_description: 歌曲描述
            
        Returns:
            tuple: (system_content, user_content)
        """
        system_content = system_prompt
        user_content = user_template.format(song_description=song_description, lyrics=lyrics)

        return system_content, user_content

    def extract_lrc_from_response(self, response_text: str) -> str:
        """从 GPT 响应中提取 LRC 内容
        
        Args:
            response_text: GPT 的响应文本
            
        Returns:
            str: 提取的 LRC 内容
            
        Raises:
            ValueError: 当无法提取有效的 LRC 内容时
        """
        # 查找 ```lrc ... ``` 包裹的内容
        lrc_pattern = r'```lrc\s*(.*?)\s*```'
        match = re.search(lrc_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if match:
            lrc_content = match.group(1).strip()
        else:
            # 如果没有找到包裹的内容，尝试直接提取 LRC 格式的行
            lines = response_text.strip().split('\n')
            lrc_lines = []
            for line in lines:
                line = line.strip()
                if re.match(r'\[\d{2}:\d{2}\.\d{2}\]', line):
                    lrc_lines.append(line)

            if lrc_lines:
                lrc_content = '\n'.join(lrc_lines)
            else:
                raise ValueError("无法从响应中提取有效的 LRC 内容")

        # 验证 LRC 格式
        if not self._validate_lrc_format(lrc_content):
            raise ValueError("提取的 LRC 格式不正确")

        return lrc_content

    def _validate_lrc_format(self, lrc_content: str) -> bool:
        """验证 LRC 格式是否正确
        
        Args:
            lrc_content: LRC 内容
            
        Returns:
            bool: 格式是否正确
        """
        lines = lrc_content.strip().split('\n')
        if not lines:
            return False

        # 检查每行是否符合 LRC 格式
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # LRC 行应该以 [mm:ss.xx] 开头
            if not re.match(r'^\[\d{2}:\d{2}\.\d{2}\]', line):
                return False

        return True

    def generate_lrc(self, lyrics: str, song_description: str, max_attempts: int = 3) -> str:
        """生成 LRC 格式的歌词时间轴
        
        Args:
            lyrics: 歌词内容
            song_description: 歌曲描述
            max_attempts: 最大尝试次数
            
        Returns:
            str: LRC 格式的内容
            
        Raises:
            ValueError: 当达到最大尝试次数仍无法生成有效 LRC 时
        """
        system_content, user_content = self.build_lrc_prompt(lyrics, song_description)

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{
                        "role": "system",
                        "content": system_content
                    }, {
                        "role": "user",
                        "content": user_content
                    }],
                    temperature=0.4
                )

                response_text = response.choices[0].message.content.strip()
                lrc_content = self.extract_lrc_from_response(response_text)

                return lrc_content

            except ValueError as e:
                if attempt == max_attempts - 1:
                    raise ValueError(f"经过 {max_attempts} 次尝试后仍无法生成有效的 LRC 格式: {e}")
                continue
        raise ValueError(f"经过 {max_attempts} 次尝试后仍无法生成有效的 LRC 格式")

    def remove_structural_tags(self, lrc_content: str) -> str:
        """去除 LRC 内容中的结构化标签
        
        Args:
            lrc_content: 包含结构化标签的 LRC 内容
            
        Returns:
            str: 去除结构化标签后的 LRC 内容
        """
        lines = lrc_content.strip().split('\n')
        filtered_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是纯结构化标签行
            # 正则规则：如果单行内容仅包含一个 [] 或者 （），并且括号内存在结构化标签，则认为该行为非歌词
            # 匹配时间戳后的内容
            timestamp_match = re.match(r'^\[\d{2}:\d{2}\.\d{2}\](.*)$', line)
            if timestamp_match:
                content_after_timestamp = timestamp_match.group(1).strip()

                # 检查时间戳后是否只有一个方括号或圆括号包围的结构化标签
                # 支持大小写变化和数字后缀（如 Verse1, Chorus2 等）
                structural_pattern = r'^[\[\(]\s*(start|end|intro|outro|break|bridge|inst|solo|verse\d*|chorus\d*)\s*[\]\)]$'

                if re.match(structural_pattern, content_after_timestamp, re.IGNORECASE):
                    continue  # 跳过结构化标签行

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def get_song_duration(self, lrc_content: str) -> int:
        """获取歌曲总时长
        
        Args:
            lrc_content: 包含结构化标签的 LRC 内容
            
        Returns:
            int: 歌曲总时长，单位为秒
            
        Raises:
            ValueError: 当最后一行不包含 [End] 标签时
        """
        lines = lrc_content.strip().split('\n')

        # 从后往前找到最后一个带有时间戳的行
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # 检查是否包含时间戳
            timestamp_match = re.match(r'^\[(\d{2}):(\d{2})\.(\d{2})\](.*)$', line)
            if timestamp_match:
                minutes = int(timestamp_match.group(1))
                seconds = int(timestamp_match.group(2))
                centiseconds = int(timestamp_match.group(3))
                content_after_timestamp = timestamp_match.group(4)

                # 检查是否包含 [End] 标签（支持大小写）
                if re.search(r'\[end\]', content_after_timestamp, re.IGNORECASE):
                    # 转换为总秒数（四舍五入）
                    total_seconds = minutes*60 + seconds + round(centiseconds / 100)
                    return total_seconds
                else:
                    raise ValueError("最后一个带有时间戳的行不包含 [End] 标签")

        raise ValueError("找不到带有时间戳的行")
