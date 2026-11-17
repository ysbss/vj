package com.ds.automation.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public final class PythonProcessSupport {

  private PythonProcessSupport() {}

  public static File projectRoot(String pythonPath) {
    return new File(pythonPath).getParentFile().getParentFile();
  }

  public static File pythonHome(String pythonPath) {
    return new File(pythonPath).getParentFile();
  }

  public static void configureProcessEnvironment(
      ProcessBuilder builder, String pythonPath, String ffmpegPath) {
    File root = projectRoot(pythonPath);
    File pythonDir = pythonHome(pythonPath);
    builder.directory(root);

    Map<String, String> env = builder.environment();
    String pathKey = env.containsKey("Path") ? "Path" : "PATH";
    List<String> pathParts = new ArrayList<>();
    pathParts.add(new File(ffmpegPath).getParent());
    pathParts.add(pythonDir.getAbsolutePath());
    pathParts.add(new File(pythonDir, "Scripts").getAbsolutePath());
    pathParts.add(new File(pythonDir, "Library" + File.separator + "bin").getAbsolutePath());
    File torchLib =
        new File(
            pythonDir,
            "Lib" + File.separator + "site-packages" + File.separator + "torch" + File.separator + "lib");
    if (torchLib.isDirectory()) {
      pathParts.add(torchLib.getAbsolutePath());
    }
    String existingPath = env.getOrDefault(pathKey, "");
    if (!existingPath.isBlank()) {
      pathParts.add(existingPath);
    }
    env.put(pathKey, String.join(File.pathSeparator, pathParts));

    env.put("PYTHONHOME", pythonDir.getAbsolutePath());
    env.put("FFMPEG_BINARY", ffmpegPath);
    env.put("PYTHONUNBUFFERED", "1");
    env.put("PYTHONIOENCODING", "utf-8");
    env.put("KMP_DUPLICATE_LIB_OK", "TRUE");
    configureHuggingFaceEnvironment(env, root);
  }

  public static void configureWhisperRuntimeEnvironment(
      ProcessBuilder builder, String whisperDevice) {
    if ("cpu".equals(whisperDevice)) {
      builder.environment().put("CUDA_VISIBLE_DEVICES", "");
    }
  }

  public static int runLoggedProcess(ProcessBuilder builder, Logger logger)
      throws IOException, InterruptedException {
    builder.redirectErrorStream(true);
    Process process = builder.start();
    try (BufferedReader reader =
        new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
      String line;
      while ((line = reader.readLine()) != null) {
        logger.info(line);
      }
    }
    return process.waitFor();
  }

  private static void configureHuggingFaceEnvironment(Map<String, String> env, File projectRoot) {
    File hfHome = new File(projectRoot, "models" + File.separator + "hf-cache");
    File hubCache = new File(hfHome, "hub");
    hfHome.mkdirs();
    hubCache.mkdirs();
    env.putIfAbsent("HF_HOME", hfHome.getAbsolutePath());
    env.putIfAbsent("HUGGINGFACE_HUB_CACHE", hubCache.getAbsolutePath());
    env.putIfAbsent("HF_ENDPOINT", "https://hf-mirror.com");
    env.putIfAbsent("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com");
    env.putIfAbsent("HF_HUB_DISABLE_SYMLINKS_WARNING", "1");
  }

  /** CPU 回退时避免 ctranslate2 再次初始化 CUDA（尤其 GPU 进程刚崩溃后） */
  public static void forceCpuOnlyEnvironment(ProcessBuilder builder) {
    builder.environment().put("CUDA_VISIBLE_DEVICES", "");
  }
}
