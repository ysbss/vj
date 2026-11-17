package com.ds.automation.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static com.ds.CommonConstants.AI_DETECT_INFER_DEVICE_SCRIPT;

public final class InferDeviceResolver {

  public record InferDevice(
      String demucsDevice, String whisperDevice, String computeType, String gpuName) {}

  private InferDeviceResolver() {}

  public static InferDevice resolve(String pythonPath, String projectRoot, Logger logger) {
    InferDevice detected = runDetectScript(pythonPath, projectRoot, logger);
    if ("cuda".equals(detected.demucsDevice()) || "cuda".equals(detected.whisperDevice())) {
      logger.info(
          ">>> GPU 加速已启用"
              + (detected.gpuName().isBlank() ? "" : "（" + detected.gpuName() + "）")
              + " | Demucs="
              + detected.demucsDevice()
              + ", Whisper="
              + detected.whisperDevice()
              + ", compute_type="
              + detected.computeType());
    } else {
      logger.info(">>> 未检测到可用 GPU，使用 CPU 推理（Demucs + Whisper）");
    }
    return detected;
  }

  private static InferDevice runDetectScript(
      String pythonPath, String projectRoot, Logger logger) {
    try {
      ProcessBuilder builder =
          new ProcessBuilder(
              List.of(
                  pythonPath,
                  new File(projectRoot, AI_DETECT_INFER_DEVICE_SCRIPT).getAbsolutePath()));
      PythonProcessSupport.configureProcessEnvironment(
          builder, pythonPath, new File(projectRoot, "ffmpeg/bin/ffmpeg.exe").getAbsolutePath());
      builder.redirectErrorStream(true);

      Process process = builder.start();
      Map<String, String> values = new HashMap<>();
      try (BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
        String line;
        while ((line = reader.readLine()) != null) {
          int idx = line.indexOf('=');
          if (idx > 0) {
            values.put(line.substring(0, idx), line.substring(idx + 1));
          }
        }
      }
      int exitCode = process.waitFor();
      if (exitCode != 0) {
        logger.warning("设备检测脚本退出码 " + exitCode + "，回退 CPU");
        return cpuFallback();
      }

      String demucsDevice = normalizeDevice(values.get("demucs_device"));
      String whisperDevice = normalizeDevice(values.get("whisper_device"));
      String computeType = values.getOrDefault("compute_type", "int8");
      String gpuName = values.getOrDefault("gpu_name", "");
      if (!"float16".equals(computeType) && !"int8".equals(computeType)) {
        computeType = "cuda".equals(whisperDevice) ? "float16" : "int8";
      }
      return new InferDevice(demucsDevice, whisperDevice, computeType, gpuName);
    } catch (Exception e) {
      logger.warning("设备检测失败，回退 CPU: " + e.getMessage());
      return cpuFallback();
    }
  }

  private static String normalizeDevice(String device) {
    return "cuda".equalsIgnoreCase(device) ? "cuda" : "cpu";
  }

  private static InferDevice cpuFallback() {
    return new InferDevice("cpu", "cpu", "int8", "");
  }
}
