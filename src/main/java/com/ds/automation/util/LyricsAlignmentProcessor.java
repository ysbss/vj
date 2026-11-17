package com.ds.automation.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static com.ds.CommonConstants.AI_ALIGN_LYRICS_SCRIPT;
import static com.ds.CommonConstants.WHISPER_CPU_COMPUTE_TYPE;
import static com.ds.CommonConstants.WHISPER_CPU_DEVICE;
import static com.ds.CommonConstants.WHISPER_CPU_RETRY_WAIT_MS;

public class LyricsAlignmentProcessor {

  private final ExecutorService lyricsAlignmentExecutor = Executors.newSingleThreadExecutor();
  private final String pythonPath;

  private final String ffmpegPath;
  private final String whisperDevice;
  private final String computeType;
  private final Logger logger;

  public LyricsAlignmentProcessor(
      String pythonPath,
      String ffmpegPath,
      String whisperDevice,
      String computeType,
      Logger logger) {
    this.pythonPath = pythonPath;
    this.ffmpegPath = ffmpegPath;
    this.whisperDevice = whisperDevice;
    this.computeType = computeType;
    this.logger = logger;
  }

  public void alignLyrics(List<String> lyrics, String audioPath) {}

  public CompletableFuture<Boolean> alignLyrics(
      File vocalsFile, File lyricsFile, File outLrc, File outTxt) {
    return CompletableFuture.supplyAsync(
        () -> alignLyricsWithFallback(vocalsFile, lyricsFile, outLrc, outTxt),
        lyricsAlignmentExecutor);
  }

  private boolean alignLyricsWithFallback(
      File vocalsFile, File lyricsFile, File outLrc, File outTxt) {
    logger.info(
        ">>> 歌词对齐设备: " + whisperDevice + " (compute_type=" + computeType + ")");
    logger.info(">>> 人声文件: " + vocalsFile.getAbsolutePath());
    if (runAlignOnce(vocalsFile, lyricsFile, outLrc, outTxt, whisperDevice, computeType, false)) {
      return true;
    }
    if (!"cuda".equals(whisperDevice)) {
      return false;
    }
    logger.warning(">>> GPU 歌词对齐失败，等待 " + WHISPER_CPU_RETRY_WAIT_MS + "ms 后改用 CPU 重试...");
    try {
      Thread.sleep(WHISPER_CPU_RETRY_WAIT_MS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return false;
    }
    boolean cpuSuccess =
        runAlignOnce(
            vocalsFile,
            lyricsFile,
            outLrc,
            outTxt,
            WHISPER_CPU_DEVICE,
            WHISPER_CPU_COMPUTE_TYPE,
            true);
    if (cpuSuccess) {
      logger.info("✅ CPU 回退歌词对齐成功");
    }
    return cpuSuccess;
  }

  private boolean runAlignOnce(
      File vocalsFile,
      File lyricsFile,
      File outLrc,
      File outTxt,
      String device,
      String computeType,
      boolean cpuFallback) {
    try {
      List<String> command =
          buildLyricsAlignmentCommand(vocalsFile, lyricsFile, outLrc, outTxt, device, computeType);
      logger.info(">>> 执行命令: " + formatCommandForLog(command));

      ProcessBuilder processBuilder = new ProcessBuilder(command);
      PythonProcessSupport.configureProcessEnvironment(processBuilder, pythonPath, ffmpegPath);
      PythonProcessSupport.configureWhisperRuntimeEnvironment(processBuilder, device);
      if (cpuFallback) {
        PythonProcessSupport.forceCpuOnlyEnvironment(processBuilder);
      }

      int exitCode = PythonProcessSupport.runLoggedProcess(processBuilder, logger);
      if (exitCode != 0) {
        logger.severe(
            "❌ 歌词对齐进程异常退出（device="
                + device
                + ", compute_type="
                + computeType
                + "），错误码: "
                + exitCode);
        return false;
      }
      return true;
    } catch (IOException | InterruptedException e) {
      logger.severe("❌ 歌词对齐进程异常退出，错误信息: " + e.getMessage());
      return false;
    }
  }

  private String formatCommandForLog(List<String> command) {
    return command.stream()
        .map(part -> part.contains(" ") ? "\"" + part + "\"" : part)
        .collect(Collectors.joining(" "));
  }

  private List<String> buildLyricsAlignmentCommand(
      File vocalsFile,
      File lyricsFile,
      File outLrc,
      File outTxt,
      String device,
      String computeType) {
    File alignScript =
        new File(PythonProcessSupport.projectRoot(pythonPath), AI_ALIGN_LYRICS_SCRIPT);
    List<String> commands = new ArrayList<>();
    commands.add(pythonPath);
    commands.add(alignScript.getAbsolutePath());
    commands.add("--vocals");
    commands.add(vocalsFile.getAbsolutePath());
    commands.add("--lyrics");
    commands.add(lyricsFile.getAbsolutePath());
    commands.add("--out-lrc");
    commands.add(outLrc.getAbsolutePath());
    commands.add("--out-txt");
    commands.add(outTxt.getAbsolutePath());
    commands.add("--device");
    commands.add(device);
    commands.add("--compute-type");
    commands.add(computeType);
    return commands;
  }

  public void shutdown() {
    lyricsAlignmentExecutor.shutdown();
  }
}
