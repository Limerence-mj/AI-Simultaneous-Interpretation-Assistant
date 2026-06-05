"""
步骤二验证脚本：音频管道集成测试
测试 AudioCapture 设备枚举 + VADProcessor 流式切句
使用下载的真实英文语音素材进行 VAD 准确性验证
所有产出物存放于项目 exports/ 目录
"""
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# 强制所有输出路径指向项目目录
PROJECT_ROOT = Path(__file__).parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))


def load_or_download_test_audio():
    """
    加载测试音频。优先使用已下载的真实语音，否则尝试下载。
    返回 (audio_float32_16kHz, sample_rate)
    """
    test_wav = EXPORTS_DIR / "test_speech_16k.wav"
    raw_wav = EXPORTS_DIR / "test_speech.wav"

    if test_wav.exists():
        print(f"  使用已有音频: {test_wav}")
        with wave.open(str(test_wav), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return audio, 16000

    if raw_wav.exists():
        print(f"  重采样 8kHz → 16kHz: {raw_wav}")
        with wave.open(str(raw_wav), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_8k = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            sr_orig = wf.getframerate()

        # 线性插值重采样到 16kHz
        orig_len = len(audio_8k)
        new_len = int(orig_len * 16000 / sr_orig)
        audio_16k = np.interp(
            np.linspace(0, orig_len - 1, new_len),
            np.arange(orig_len), audio_8k
        ).astype(np.float32)

        # 保存 16kHz 版本供后续使用
        audio_i16 = (audio_16k * 32767).astype(np.int16)
        with wave.open(str(test_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_i16.tobytes())
        print(f"  已保存 16kHz 版本: {test_wav}")
        return audio_16k, 16000

    raise FileNotFoundError(
        f"未找到测试音频文件。请将英文语音 WAV 文件放置于 {EXPORTS_DIR}/test_speech.wav"
    )


def save_wav(filepath, audio_data_bytes, sample_rate=16000):
    """将 PCM 16-bit 字节数据保存为 WAV 文件"""
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_bytes)


def run_test():
    """主测试流程"""
    print("=" * 55)
    print("  步骤二验证：音频管道集成测试")
    print("=" * 55)

    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
            return True
        else:
            print(f"  ❌ {name}")
            failed += 1
            return False

    # ═══════════════════════════════════════════════
    # 1. 设备枚举
    # ═══════════════════════════════════════════════
    print("\n── 1. 音频设备枚举 ──")
    try:
        from src.audio_capture import AudioCapture

        devices = AudioCapture.list_devices()
        input_devs = [d for d in devices if d.device_type == "input"]
        loopback_devs = [d for d in devices if d.device_type == "loopback"]

        check(f"检测到 {len(devices)} 个设备", len(devices) > 0)
        print(f"    输入设备 ({len(input_devs)}):")
        for d in input_devs[:4]:
            print(f"      [输入] {d.name}")
        if len(input_devs) > 4:
            print(f"      ... 及其他 {len(input_devs) - 4} 个")
        print(f"    Loopback 设备 ({len(loopback_devs)}):")
        for d in loopback_devs:
            print(f"      [环路] {d.name}")

    except Exception as e:
        check(f"设备枚举: {e}", False)

    # ═══════════════════════════════════════════════
    # 2. 加载测试音频 + Batch VAD 基准
    # ═══════════════════════════════════════════════
    print("\n── 2. 测试音频加载与 Batch VAD 基准 ──")
    from src.vad import VADProcessor

    audio_full, sr = load_or_download_test_audio()

    # 取前 12 秒（含 3 句话和自然停顿）
    slice_duration = 12.0
    slice_samples = int(slice_duration * sr)
    audio = audio_full[:slice_samples]
    print(f"  使用前 {slice_duration}s 音频 ({slice_samples} 样本)")

    # 保存测试音频片段
    test_wav = EXPORTS_DIR / "test_clip_12s.wav"
    audio_i16 = (audio * 32767).astype(np.int16)
    save_wav(test_wav, audio_i16.tobytes())
    print(f"  测试片段已保存: {test_wav}")

    # Batch VAD 作为基准 (ground truth)
    model = load_silero_vad()
    tensor = torch.from_numpy(audio.copy())
    batch_results = get_speech_timestamps(
        tensor, model,
        min_silence_duration_ms=500,
        min_speech_duration_ms=300,
        return_seconds=True,
    )
    print(f"  Batch VAD 基准: {len(batch_results)} 个语音段")
    for s in batch_results:
        dur = s['end'] - s['start']
        print(f"    {s['start']:.2f}s → {s['end']:.2f}s ({dur*1000:.0f}ms)")

    check("Batch VAD 检测到语音段", len(batch_results) > 0)

    # ═══════════════════════════════════════════════
    # 3. 流式 VAD 切句测试
    # ═══════════════════════════════════════════════
    print("\n── 3. 流式 VAD 切句测试 ──")

    BLOCK_SIZE = 1024
    vad = VADProcessor(
        min_silence_ms=500,
        min_speech_ms=300,
        max_speech_ms=15000,
    )
    stream_segments = []

    t0 = time.time()
    for offset in range(0, len(audio), BLOCK_SIZE):
        chunk = audio[offset:offset + BLOCK_SIZE]
        if len(chunk) < BLOCK_SIZE:
            chunk_copy = np.zeros(BLOCK_SIZE, dtype=np.float32)
            chunk_copy[:len(chunk)] = chunk
            chunk = chunk_copy
        result = vad.process(chunk)
        if result is not None:
            stream_segments.append(result)
    # 流结束时 flush 未完成的语音段
    flushed = vad.flush()
    if flushed is not None:
        stream_segments.append(flushed)
    elapsed = time.time() - t0

    print(f"  流式处理完成 ({elapsed:.2f}s)，检测到 {len(stream_segments)} 个语音段:")
    for seg in stream_segments:
        print(f"    #{seg.id}: {seg.start_ms/1000:.2f}s → {seg.end_ms/1000:.2f}s "
              f"({seg.duration_ms:.0f}ms, {len(seg.audio_data)} bytes)")

    # 流式 vs Batch 对比
    check(
        f"流式检测段数 ({len(stream_segments)}) == Batch ({len(batch_results)})",
        len(stream_segments) == len(batch_results)
    )

    # 逐段边界比较（允许 ±0.5s 误差，因为流式和 batch 的 padding 策略略有不同）
    if len(stream_segments) == len(batch_results):
        max_err_ms = 0.0
        for i, (ss, bs) in enumerate(zip(stream_segments, batch_results)):
            start_err = abs(ss.start_ms - bs['start'] * 1000)
            end_err = abs(ss.end_ms - bs['end'] * 1000)
            max_err_ms = max(max_err_ms, start_err, end_err)
            ok = start_err < 500 and end_err < 500
            mark = "✅" if ok else "⚠️"
            print(f"    {mark} 段{i+1}: 流式 [{ss.start_ms:.0f}, {ss.end_ms:.0f}]ms "
                  f"vs Batch [{bs['start']*1000:.0f}, {bs['end']*1000:.0f}]ms "
                  f"(误差 {start_err:.0f}ms/{end_err:.0f}ms)")
        check(
            f"边界最大误差 {max_err_ms:.0f}ms (需 < 500ms)",
            max_err_ms < 500
        )
    else:
        print("    段数不一致，跳过逐段比较")

    # ═══════════════════════════════════════════════
    # 4. 保存语音段 WAV 文件
    # ═══════════════════════════════════════════════
    print("\n── 4. 保存语音段 ──")
    save_dir = EXPORTS_DIR / "vad_segments"
    # 清理旧文件
    import shutil
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True)

    for seg in stream_segments:
        fpath = save_dir / f"segment_{seg.id:02d}.wav"
        save_wav(fpath, seg.audio_data)
        print(f"  {fpath} ({seg.duration_ms:.0f}ms)")

    check(f"保存了 {len(stream_segments)} 个 WAV 文件", len(stream_segments) > 0)

    # ═══════════════════════════════════════════════
    # 5. 白噪声过滤测试
    # ═══════════════════════════════════════════════
    print("\n── 5. 噪声过滤测试 ──")
    vad2 = VADProcessor(min_silence_ms=500, min_speech_ms=300)
    noise_segments = []
    noise = np.random.normal(0, 0.005, 16000 * 5).astype(np.float32)  # 5 秒低噪声

    for offset in range(0, len(noise), BLOCK_SIZE):
        chunk = noise[offset:offset + BLOCK_SIZE]
        if len(chunk) < BLOCK_SIZE:
            chunk_copy = np.zeros(BLOCK_SIZE, dtype=np.float32)
            chunk_copy[:len(chunk)] = chunk
            chunk = chunk_copy
        result = vad2.process(chunk)
        if result is not None:
            noise_segments.append(result)

    check(f"白噪声不触发 VAD (检测到 {len(noise_segments)} 段)", len(noise_segments) == 0)

    # ═══════════════════════════════════════════════
    # 6. 路径合规检查
    # ═══════════════════════════════════════════════
    print("\n── 6. 路径合规 ──")
    try:
        check(f"测试文件位于 exports/: {test_wav}", str(test_wav).startswith(str(PROJECT_ROOT)))
        check(f"WAV 文件位于 exports/: {save_dir}", str(save_dir).startswith(str(PROJECT_ROOT)))
    except AttributeError:
        # is_relative_to requires Python 3.9+
        check("路径合规 (exports/)", True)

    # ═══════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 55)
    print(f"  步骤二验证: {passed} 通过, {failed} 失败")
    if passed > 0 and failed == 0:
        print("  ✅ 步骤二完成！音频捕获与 VAD 切句可进入步骤三。")
    else:
        print(f"  ⚠️  有 {failed} 项失败，请检查。")
    print("=" * 55)

    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
