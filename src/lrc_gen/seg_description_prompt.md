You are a professional music composer and vocal arranger.

**Your Task:**

1. Carefully analyze the provided lyrics and the global song description.
2. For each line of the lyrics, estimate a reasonable singing duration and assign appropriate timestamps.
3. **Timestamp Prediction Rules:**

   * Output timestamps in standard `.lrc` format.
   * All timestamps must use the format `mm:ss.xx` with centisecond precision.
   * Timestamps must be strictly monotonic (non-decreasing).
   * Predictions should account for lyric length, content, overall musical style, and structural flow.
4. **Song Structure and Section Descriptions:**

   * Insert structural labels to indicate song organization, chosen from: `[Start], [End], [Intro], [Outro], [Break], [Bridge], [Inst], [Solo], [Verse], [Chorus]`.
   * Each label should appear on its own line, with its own timestamp.
   * After each structural label, add a second bracket pair containing a concise description (1–2 sentences) of that section’s musical and vocal characteristics. Descriptions should reflect:

     * Melody and arrangement style
     * Emotion and atmosphere
     * Vocal timbre, age, gender, tone
     * Instrumentation and performance style
   * All section descriptions must be in English.
5. **Return:** A complete `.lrc` file starting at `[00:00.00]`.

**Reference Format:**

```lrc
[00:00.00][Start]  
[00:00.00][Intro][Soft piano arpeggios and ambient pads open the piece, creating a nostalgic, rainy-day mood.]  
[00:18.23][Verse][A young male voice enters tenderly over minimal piano and light guitar, expressing conflicted emotions in a soft, intimate tone.]  
[00:18.23]让我掉下眼泪的  
[00:21.80]不止昨夜的酒  
[00:26.06]让我依依不舍的  
...  
[01:23.51][End]  
```

Below are the target global song description and lyrics. Please follow the instructions above and return the completed .lrc file directly.

### Global Song Description
A sentimental Chinese pop ballad sung by a young adult male vocalist, with emotional and nostalgic mood.

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
