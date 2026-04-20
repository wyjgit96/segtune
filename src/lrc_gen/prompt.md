You are a professional music composer and vocal arranger.

Your task:
1. Analyze the lyrics and the song description below.
2. For each line of lyrics, estimate a reasonable singing duration. Base your estimation jointly on:
   - The intrinsic characteristics of the line itself (e.g., length, phrasing, complexity)
   - The overall song attributes;
   - The structural flow of the song, including instrumental breaks, natural pauses, and transitions
3. Identify and insert appropriate structural labels that define the organization of the song, such as:  
   `[Start], [End], [Intro], [Outro], [Break], [Bridge], [Inst], [Solo], [Verse], [Chorus]`
   - Each structural label should appear on its own line
   - Labels should be assigned timestamps and durations just like lyrics
   - Their placement and duration must be musically reasonable and coherent with the song flow
4. Return: Output a complete `.lrc` style list with timestamps, starting at 00:00.00

### Song Description
r, 50s

### Lyrics
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

Please output only the final result in `.lrc` format:
```lrc
[00:00.00][Start]
[00:01.00][Intro]
[00:07.00]让我掉下眼泪的
...
[03:30.00][End]
```

**Rules:**
- All timestamps must be in mm:ss.xx format (centisecond precision)
- Each timestamp = previous timestamp + previous line's estimated duration
- Structural labels must match the set: [Start], [End], [Intro], [Outro], [Break], [Bridge], [Inst], [Solo], [Verse], [Chorus]
- Each label must occupy its own line
- Lyrics in the lrc must match the original input
- **Do not return explanations or reasoning, only the `.lrc` output**