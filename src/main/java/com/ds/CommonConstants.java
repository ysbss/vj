package com.ds;

import java.util.List;

public interface CommonConstants {
  String DEMUCS_MODEL = "htdemucs_ft";

  /** Demucs 结束后等待 GPU 释放显存，再启动 Whisper */
  int POST_DEMucs_WAIT_MS = 5000;

  String WHISPER_CPU_DEVICE = "cpu";
  String WHISPER_CPU_COMPUTE_TYPE = "int8";

  /** GPU 崩溃后等待驱动恢复，再 CPU 重试 */
  int WHISPER_CPU_RETRY_WAIT_MS = 5000;

  String DOUBAO_ENDPOINT = "";

  String AI_ALIGN_LYRICS_SCRIPT = "scripts/align_lyrics.py";

  String AI_DETECT_INFER_DEVICE_SCRIPT = "scripts/detect_infer_device.py";

  List<String> SUPPORTED_MEDIA_EXTENSIONS =
      List.of(
          ".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v", ".m4a",
              ".aac", ".wav", ".ogg", ".flac", ".mp3");
}
