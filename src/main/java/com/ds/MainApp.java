/*
 * VJ - Batch video vocal removal tool.
 * Copyright (c) 2026 yusu
 * SPDX-License-Identifier: MIT
 */
package com.ds;

import com.ds.automation.util.AiSeparationProcessor;
import com.ds.automation.util.InferDeviceResolver;
import com.ds.automation.util.InferDeviceResolver.InferDevice;
import com.ds.automation.util.LyricsAlignmentProcessor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import static com.ds.CommonConstants.*;

public class MainApp {
  private static final Logger logger = Logger.getLogger("VjLogger");

  //  public static void main(String[] args) {
  //    // 核心逻辑：动态获取当前运行的 .exe 或 .jar 所在的绝对根目录
  //    String currentRootPath = "";
  //    try {
  //      String path = MainApp.class.getProtectionDomain().getCodeSource().getLocation().getPath();
  //      // 防止路径中含有中文或空格导致乱码，进行 URL 解码
  //      currentRootPath = URLDecoder.decode(path, StandardCharsets.UTF_8);
  //
  //      // 转换为标准的 File 对象，并获取其父级目录（即程序运行的根目录）
  //      File jarFile = new File(currentRootPath);
  //      currentRootPath = jarFile.getParentFile().getAbsolutePath();
  //    } catch (Exception e) {
  //      // 兜底策略：如果获取失败，尝试基于系统属性获取当前工作目录
  //      currentRootPath = System.getProperty("user.dir");
  //    }
  //
  //    // ====================================================================
  //    // 【关键设计】根据现实环境，动态拼接 ffmpeg 路径
  //    // ====================================================================
  //    String ffmpegPath;
  //
  //    // 我们可以做一个小开关，方便你在本地开发和最终发布之间无缝切换
  //    boolean isDevelopmentMode = true; // 在你自己电脑上开发测试时设为 true
  //
  //    if (isDevelopmentMode) {
  //      // 1. 本地开发模式：指向你今天刚下载解压的绝对路径
  //      ffmpegPath = "D:\\tools\\ffmpeg\\bin\\ffmpeg.exe";
  //    } else {
  //      // 2. 生产交付模式：动态寻找跟着安装包一起走、躺在同级 bin 目录下的 ffmpeg
  //      // 对应结构：你的根目录/bin/ffmpeg.exe
  //      ffmpegPath = currentRootPath + File.separator + "bin" + File.separator + "ffmpeg.exe";
  //    }
  //
  //    logger.info("====== 当前系统识别到的 FFmpeg 路径为: " + ffmpegPath);
  //
  //    // 后面继续初始化你的 VideoAudioProcessor...
  //    // VideoAudioProcessor processor = new VideoAudioProcessor(ffmpegPath);
  //  }
  public static void main(String[] args) {
    try {
      runPipeline();
    } finally {
      waitForExit();
    }
  }

  private static void runPipeline() {
    String rootPath = resolveAppRoot();
    initLogging(new File(rootPath));
    logger.info("应用根目录: " + rootPath);
    String ffmpegPath =
        rootPath
            + File.separator
            + "ffmpeg"
            + File.separator
            + "bin"
            + File.separator
            + "ffmpeg.exe";
    String pythonPath = rootPath + File.separator + "python" + File.separator + "python.exe";

    File inputDir = new File(rootPath, "input_objects");
    File outputObjectsDir = new File(rootPath, "output_objects");
    outputObjectsDir.mkdirs();

    // 2. 批量扫描：获取 input_objects 下是否存在子目录 如果没有提示需要新建
    if (!inputDir.exists()) {
      logger.warning("⚠️ input_objects 目录不存在，请新建后再运行！");
      return;
    }
    File[] subDirs = inputDir.listFiles((dir, name) -> dir.isDirectory());
    if (subDirs == null || subDirs.length == 0) {
      logger.warning("⚠️ input_objects 目录内没有子目录，请新建后再运行！");
      return;
    }
    InferDevice inferDevice =
        InferDeviceResolver.resolve(pythonPath, rootPath, logger);
    AiSeparationProcessor aiProcessor =
        new AiSeparationProcessor(
            pythonPath, ffmpegPath, inferDevice.demucsDevice(), logger);
    LyricsAlignmentProcessor lyricsAlignmentProcessor =
        new LyricsAlignmentProcessor(
            pythonPath,
            ffmpegPath,
            inferDevice.whisperDevice(),
            inferDevice.computeType(),
            logger);

    for (File subDir : subDirs) {
      logger.info("\n🎬 正在处理任务: " + subDir.getName());
      File sourceMedia = resolveSourceMedia(subDir);
      if (sourceMedia == null) {
        logger.severe("⚠️ 任务目录内未找到支持的媒体文件，跳过: " + subDir.getAbsolutePath());
        continue;
      }
      File tempDir = new File(outputObjectsDir, subDir.getName().replaceFirst("\\.[^.]+$", ""));
      tempDir.mkdirs();

      try {
        // 使用 join() 阻塞当前线程，直到当前视频处理完全结束
        boolean success =
            runSingleTaskSync(
                aiProcessor,
                lyricsAlignmentProcessor,
                sourceMedia,
                tempDir,
                resolveLyricsFile(subDir),
                resolveLyricsLrcFile(tempDir),
                resolveLyricsTxtFile(tempDir));
        if (success) {
          //          System.out.println("✅ 任务完成: " + sourceVideo.getName());
          logger.info("✅ 任务完成: " + subDir.getName());
        } else {
          //          System.out.println("❌ 任务失败: " + sourceVideo.getName());
          logger.severe("❌ 任务失败: " + subDir.getName());
        }
      } catch (Exception e) {
        //        System.out.println("🚨 任务执行异常: " + e.getMessage());
        logger.severe("🚨 任务执行异常: " + e.getMessage());
      }
    }
    shutdownAll(aiProcessor, lyricsAlignmentProcessor);
    //    System.out.println("🏁 全部任务已处理完毕，流水线安全关闭。");
    logger.info("🏁 全部任务已处理完毕，流水线安全关闭。");
  }

  private static void initLogging(File appRoot) {
    logger.setUseParentHandlers(false);
    logger.setLevel(Level.INFO);

    ConsoleHandler console = new ConsoleHandler();
    console.setLevel(Level.INFO);
    console.setFormatter(new SimpleFormatter());
    logger.addHandler(console);

    try {
      File logFile = new File(appRoot, "app.log");
      FileHandler file = new FileHandler(logFile.getAbsolutePath(), true);
      file.setLevel(Level.INFO);
      file.setFormatter(new SimpleFormatter());
      logger.addHandler(file);
    } catch (IOException e) {
      System.err.println("❌ 日志文件初始化失败: " + e.getMessage());
    }
  }

  private static void waitForExit() {
    System.out.println();
    System.out.println("按回车键退出...");
    try {
      if (System.console() != null) {
        System.console().readLine();
      } else {
        Scanner scanner = new Scanner(System.in);
        scanner.nextLine();
      }
    } catch (Exception ignored) {
      // 无控制台时静默退出
    }
  }

  private static String resolveAppRoot() {
    File root = null;
    try {
      String path =
          MainApp.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
      File codeFile = new File(path);
      root = codeFile.isFile() ? codeFile.getParentFile() : codeFile;
      // jpackage：jar 在 app/ 子目录，资源与 VJ.exe 在上一级
      if ("app".equals(root.getName())) {
        root = root.getParentFile();
      }
    } catch (Exception ignored) {
      // fall through to marker / user.dir fallback
    }

    if (root != null && hasRuntimeLayout(root)) {
      return root.getAbsolutePath();
    }

    File cwd = new File(System.getProperty("user.dir"));
    if (hasRuntimeLayout(cwd)) {
      return cwd.getAbsolutePath();
    }

    return root != null ? root.getAbsolutePath() : cwd.getAbsolutePath();
  }

  /** 项目根目录应同时包含 ffmpeg 与 python */
  private static boolean hasRuntimeLayout(File dir) {
    return new File(dir, "ffmpeg/bin/ffmpeg.exe").isFile()
        && new File(dir, "python/python.exe").isFile();
  }

  private static boolean runSingleTaskSync(
      AiSeparationProcessor aiProcessor,
      LyricsAlignmentProcessor lyricsAlignmentProcessor,
      File sourceMedia,
      File tempDir,
      File lyricsFile,
      File lyricsLrcFile,
      File lyricsTxtFile) {
    File noVocalsAudio = resolveNoVocalsFlac(tempDir, sourceMedia);
    File vocalsAudio = resolveVocalsFlac(tempDir, sourceMedia);

    // 使用 join() 将异步变同步
    try {
      boolean aiSuccess = aiProcessor.separateVocal(sourceMedia, tempDir).join();
      if (!aiSuccess) return false;
      logger.info("✅【节点一成功】AI 已从媒体中剥离人声！");

      waitForFileRelease(POST_DEMucs_WAIT_MS);

      if (lyricsFile != null && lyricsFile.isFile() && vocalsAudio.isFile()) {
        logger.info(">>> 发现歌词文件，开始歌词对齐: " + lyricsFile.getAbsolutePath());
        boolean alignLyricsSuccess =
            lyricsAlignmentProcessor
                .alignLyrics(vocalsAudio, lyricsFile, lyricsLrcFile, lyricsTxtFile)
                .join();
        if (alignLyricsSuccess) {
          logger.info("✅【节点二成功】歌词对齐完成！");
          logger.info("✅ 歌词三语言lrc文件: " + lyricsLrcFile.getAbsolutePath());
          logger.info("✅ 歌词文本文件: " + lyricsTxtFile.getAbsolutePath());
        } else {
          logger.severe(
              "❌【节点二失败】歌词对齐失败，请检查网络/Whisper 模型下载或歌词文件是否正确");
        }
      } else if (lyricsFile == null || !lyricsFile.isFile()) {
        logger.info("ℹ️ 未找到 .txt 歌词文件，跳过歌词对齐");
      } else {
        logger.warning("ℹ️ 人声文件不存在，跳过歌词对齐: " + vocalsAudio.getAbsolutePath());
      }
      if (!copyAudioToOutputDir(vocalsAudio, tempDir)
          || !copyAudioToOutputDir(noVocalsAudio, tempDir)) {
        logger.severe("❌ 复制音频文件失败");
        return false;
      }
      logger.info("✅ 人声文件: " + new File(tempDir, vocalsAudio.getName()).getAbsolutePath());
      logger.info("✅ 伴奏文件: " + new File(tempDir, noVocalsAudio.getName()).getAbsolutePath());
      cleanUpTrash(new File(tempDir, DEMUCS_MODEL));
      return true;
    } catch (Exception e) {
      logger.severe("单任务执行异常: " + e.getMessage());
    }
    return false;
  }

  private static void waitForFileRelease(int millis) throws InterruptedException {
    Thread.sleep(millis);
  }

  /** Demucs 输出规则：{outputDir}/{媒体主文件名}-no_vocals.flac */
  private static File resolveNoVocalsFlac(File outputDir, File sourceMedia) {
    String baseName = sourceMedia.getName().replaceFirst("\\.[^.]+$", "");
    return new File(outputDir, DEMUCS_MODEL + File.separator + baseName + "-no_vocals.flac");
  }

  /** Demucs 输出规则：{outputDir}/{媒体主文件名}-vocals.flac */
  private static File resolveVocalsFlac(File outputDir, File sourceMedia) {
    String baseName = sourceMedia.getName().replaceFirst("\\.[^.]+$", "");
    return new File(outputDir, DEMUCS_MODEL + File.separator + baseName + "-vocals.flac");
  }

  /** 任务目录内任意 .txt 作为歌词；多个时按文件名取第一个并打日志 */
  private static File resolveLyricsFile(File taskDir) {
    File[] files = taskDir.listFiles();
    if (files == null) {
      return null;
    }
    File chosen = null;
    int txtCount = 0;
    for (File file : files) {
      if (file.isFile() && file.getName().toLowerCase().endsWith(".txt")) {
        txtCount++;
        if (chosen == null || file.getName().compareToIgnoreCase(chosen.getName()) < 0) {
          chosen = file;
        }
      }
    }
    if (txtCount > 1 && chosen != null) {
      logger.warning(
          "任务目录内有多个 .txt，使用: " + chosen.getName() + " （目录: " + taskDir.getAbsolutePath() + "）");
    }
    return chosen;
  }

  /** 在 input_objects/{任务名}/ 下查找第一个支持的媒体文件 */
  private static File resolveSourceMedia(File taskDir) {
    File[] files = taskDir.listFiles();
    if (files == null) {
      return null;
    }
    int mediaCount = 0;
    File chosen = null;
    for (String ext : SUPPORTED_MEDIA_EXTENSIONS) {
      for (File file : files) {
        if (file.isFile() && file.getName().toLowerCase().endsWith(ext)) {
          mediaCount++;
          if (chosen == null || file.getName().compareToIgnoreCase(chosen.getName()) < 0) {
            chosen = file;
          }
        }
      }
    }
    if (mediaCount > 1 && chosen != null) {
      logger.warning(
          "任务目录内有多个媒体文件，将使用: " + chosen.getName() + " （目录: " + taskDir.getAbsolutePath() + "）");
    }
    return chosen;
  }

  private static File resolveLyricsLrcFile(File outputLrcDir) {
    String baseName = outputLrcDir.getName().replaceFirst("\\.[^.]+$", "");
    return new File(outputLrcDir, baseName + "-out_lrc.lrc");
  }

  private static File resolveLyricsTxtFile(File outputTxtDir) {
    String baseName = outputTxtDir.getName().replaceFirst("\\.[^.]+$", "");
    return new File(outputTxtDir, baseName + "-out_txt.txt");
  }

  private static boolean copyAudioToOutputDir(File audioFile, File outputDir) {
    File targetFile = new File(outputDir, audioFile.getName());
    try {
      Files.copy(audioFile.toPath(), targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException e) {
      logger.severe("复制音频文件失败: " + e.getMessage());
      return false;
    }
    return true;
  }

  private static void cleanUpTrash(File demucsTrackDir) {
    if (demucsTrackDir == null || !demucsTrackDir.exists()) {
      return;
    }
    File[] files = demucsTrackDir.listFiles();
    if (files != null) {
      for (File file : files) {
        file.delete();
      }
    }
    demucsTrackDir.delete();
    //    System.out.println("🧹 [垃圾回收] Demucs 中间产物已清理。");
    logger.info("🧹 [垃圾回收] Demucs 中间产物已清理。");
  }

  private static void shutdownAll(AiSeparationProcessor ap, LyricsAlignmentProcessor lap) {
    ap.shutdown();
    lap.shutdown();
  }
}
