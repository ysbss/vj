/*
 * FFmpeg wrapper for audio extraction and video merging.
 * Copyright (c) 2026 yusu
 * SPDX-License-Identifier: MIT
 */
package com.ds.automation.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

public class VideoAudioProcessor {
  private static final int DELETE_RETRY_COUNT = 5;
  private static final int DELETE_RETRY_MS = 200;
  // 创建专属线程池，防止大文件 IO 阻塞主程序（大小可根据现实业务调整）
  private final ExecutorService executor = Executors.newFixedThreadPool(2);

  // 现实业务中，从打包目录或配置文件动态获取 ffmpeg.exe 的绝对路径
  private final String ffmpegPath;
  private final Logger logger;

  public VideoAudioProcessor(String ffmpegPath, Logger logger) {
    this.ffmpegPath = ffmpegPath;
    this.logger = logger;
  }

  /**
   * 核心业务方法：从视频中提取纯音频 (WAV 格式)
   *
   * @param videoInput 目标视频文件
   * @param audioOutput 生成的音频目标路径
   * @return 异步响应，便于前端界面监听完成状态
   */
  public CompletableFuture<Boolean> extractAudio(File videoInput, File audioOutput) {
    return CompletableFuture.supplyAsync(
        () -> {
          List<String> command = new ArrayList<>();
          command.add(ffmpegPath);
          command.add("-i");
          command.add(videoInput.getAbsolutePath());
          command.add("-vn"); // 禁用视频流
          command.add("-acodec");
          command.add("pcm_s16le"); // 指定高保真无损 WAV 编码
          command.add("-y"); // 覆盖已有文件
          command.add(audioOutput.getAbsolutePath());

          return runProcess(command, "音频提取");
        },
        executor);
  }

  /**
   * 核心业务方法：将静音的原视频与处理好的伴奏音频合成新视频
   *
   * @param originalVideo 原视频（用于提取视频流）
   * @param bgmAudio 纯伴奏音频
   * @param videoOutput 最终输出的纯伴奏视频
   * @return 异步响应
   */
  public CompletableFuture<Boolean> mergeVideoAndAudio(
      File originalVideo, File bgmAudio, File videoOutput) {
    return CompletableFuture.supplyAsync(
        () -> {
          Path tempOut = null;
          boolean committed = false;
          try {
            Path finalOut = videoOutput.toPath();
            ensureWritableParent(finalOut);

            // 扩展名必须是 .mp4，否则 FFmpeg 无法推断 muxer 格式（.part 会报 -22）
            tempOut = Files.createTempFile(finalOut.getParent(), "merging_", ".mp4");
            logger.info("合成临时文件: " + tempOut);

            if (!runProcess(buildMergeCommand(originalVideo, bgmAudio, tempOut.toFile()), "音视频合成")) {
              return false;
            }

            replaceOutputFile(tempOut, finalOut);
            committed = true;
            logger.info("成品已落盘: " + finalOut);
            return true;
          } catch (Exception e) {
            logger.severe("音视频合成异常: " + e.getMessage());
            e.printStackTrace();
            return false;
          } finally {
            if (!committed && tempOut != null) {
              try {
                Files.deleteIfExists(tempOut);
              } catch (IOException e) {
                logger.warning("清理临时合成文件失败: " + tempOut);
              }
            }
          }
        },
        executor);
  }

  private List<String> buildMergeCommand(File originalVideo, File bgmAudio, File outputFile) {
    List<String> command = new ArrayList<>();
    command.add(ffmpegPath);
    command.add("-i");
    command.add(originalVideo.getAbsolutePath());
    command.add("-i");
    command.add(bgmAudio.getAbsolutePath());
    command.add("-map");
    command.add("0:v:0");
    command.add("-map");
    command.add("1:a:0");
    command.add("-c:v");
    command.add("copy");
    command.add("-c:a");
    command.add("aac");
    command.add("-b:a");
    command.add("256k");
    command.add("-movflags");
    command.add("+faststart");
    command.add("-shortest");
    command.add("-f");
    command.add("mp4");
    command.add("-y");
    command.add(outputFile.getAbsolutePath());
    return command;
  }

  private void ensureWritableParent(Path outputPath) throws IOException {
    Path parent = outputPath.getParent();
    if (parent == null) {
      throw new IOException("输出路径无父目录: " + outputPath);
    }
    Files.createDirectories(parent);
    if (!Files.isWritable(parent)) {
      throw new IOException("输出目录不可写: " + parent);
    }
  }

  private void replaceOutputFile(Path tempOut, Path finalOut)
      throws IOException, InterruptedException {
    if (Files.exists(finalOut)) {
      File existing = finalOut.toFile();
      existing.setWritable(true);
      boolean deleted = false;
      for (int i = 0; i < DELETE_RETRY_COUNT; i++) {
        if (existing.delete()) {
          deleted = true;
          break;
        }
        Thread.sleep(DELETE_RETRY_MS);
      }
      if (!deleted && Files.exists(finalOut)) {
        throw new IOException("无法替换已存在的输出文件（可能被占用）: " + finalOut);
      }
    }

    try {
      Files.move(
          tempOut,
          finalOut,
          StandardCopyOption.REPLACE_EXISTING,
          StandardCopyOption.ATOMIC_MOVE);
    } catch (IOException atomicMoveFailed) {
      Files.move(tempOut, finalOut, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  /** 底层物理进程调度，解决 FFmpeg 阻塞缓冲区的大坑 */
  private boolean runProcess(List<String> command, String taskName) {
    try {
      ProcessBuilder builder = new ProcessBuilder(command);
      builder.redirectErrorStream(true); // 将错误流重定向到标准输出，方便统一读取
      Process process = builder.start();

      // 必须启动异步线程去消耗 FFmpeg 的输出日志，否则缓冲区满了进程会无限卡死
      try (BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
        String line;
        while ((line = reader.readLine()) != null) {
//           这里可以解析 line 中的 "time=" 来计算进度条，后续做 UI 时非常有用
//           System.out.println("[" + taskName + "] " + line);
           logger.info("[" + taskName + "] " + line);
        }
      }

      int exitCode = process.waitFor();
      if (exitCode == 0) {
//        System.out.println("====== " + taskName + " 成功 ======");
        logger.info("====== " + taskName + " 成功 ======");
        return true;
      } else {
//        System.err.println("====== " + taskName + " 失败，错误码: " + exitCode + " ======");
        logger.severe("====== " + taskName + " 失败，错误码: " + exitCode + " ======");
        return false;
      }
    } catch (Exception e) {
      e.printStackTrace();
      return false;
    }
  }

  /** 优雅关闭线程池 */
  public void shutdown() {
    executor.shutdown();
  }
}
