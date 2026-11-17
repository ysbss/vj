/*
 * Demucs AI vocal separation process launcher.
 * Copyright (c) 2026 yusu
 * SPDX-License-Identifier: MIT
 */
package com.ds.automation.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

import static com.ds.CommonConstants.DEMUCS_MODEL;

public class AiSeparationProcessor {
  // 专属线程池，用来跑极度消耗 CPU 的 AI 物理进程
  private final ExecutorService aiExecutor = Executors.newSingleThreadExecutor();

  // 你本地 Python 的绝对路径
  private final String pythonPath;

  private final String ffmpegPath;
  private final String demucsDevice;
  private final Logger logger;

  public AiSeparationProcessor(
      String pythonPath, String ffmpegPath, String demucsDevice, Logger logger) {
    this.pythonPath = pythonPath;
    this.ffmpegPath = ffmpegPath;
    this.demucsDevice = demucsDevice;
    this.logger = logger;
  }

  /**
   * 异步唤醒 AI 引擎，分离人声与伴奏
   *
   * @param mediaInput 输入媒体文件（视频或音频，Demucs 通过 FFmpeg 解码）
   * @param outputDir 期望 AI 吐出结果的根目录（建议直接和视频放一起）
   */
  public CompletableFuture<Boolean> separateVocal(File mediaInput, File outputDir) {
    return CompletableFuture.supplyAsync(
        () -> {
          List<String> command = new ArrayList<>();

          command.add(pythonPath);
          command.add("-m");
          command.add("demucs");
          command.add("--two-stems");
          command.add("vocals");
          command.add("-n");
          command.add(DEMUCS_MODEL);
          command.add("--flac");
          command.add("-d");
          command.add(demucsDevice);
          command.add("-o");
          command.add(outputDir.getAbsolutePath());
          command.add("--filename");
          command.add("{track}-{stem}.{ext}");
          command.add(mediaInput.getAbsolutePath());

          //          System.out.println(">>> [AI 核心] 正在全力燃烧 CPU 提取伴奏，此过程较慢，请耐心等待...");
          logger.info(
              ">>> [AI 核心] 正在使用 "
                  + demucsDevice.toUpperCase()
                  + " 分离人声，请耐心等待...");
          return runAiProcess(command);
        },
        aiExecutor);
  }

  private boolean runAiProcess(List<String> command) {
    try {
      ProcessBuilder builder = new ProcessBuilder(command);
      PythonProcessSupport.configureProcessEnvironment(builder, pythonPath, ffmpegPath);
      builder.redirectErrorStream(true);

      Process process = builder.start();

      // 实时打印 AI 算矩阵的进度日志（百分比）
      try (BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
        String line;
        while ((line = reader.readLine()) != null) {
          // Demucs 运行时会不断刷新打印进度，我们在控制台实时盯着它
          if (line.contains("Separating") || line.contains("%")) {
            //            System.out.println("[AI 实时进度] " + line.trim());
            logger.info("[AI 实时进度] " + line.trim());
          }
        }
      }

      int exitCode = process.waitFor();
      if (exitCode != 0) {
        logger.severe("❌ Demucs 进程异常退出，错误码: " + exitCode);
        //        System.err.println("❌ Demucs 进程异常退出，错误码: " + exitCode);
      }
      return exitCode == 0;

    } catch (Exception e) {
      logger.severe("❌ Demucs 进程异常退出，错误信息: " + e.getMessage());
      return false;
    }
  }

  public void shutdown() {
    aiExecutor.shutdown();
  }
}
