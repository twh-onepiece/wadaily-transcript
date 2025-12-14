# Realtime Speech Transcription Server

日本語音声をリアルタイムで文字起こしするWebSocketサーバー。kotoba-whisper-v2.0を使用。

## 使い方

### ビルド
```bash
docker build -t wadaily.sakuracr.jp/wadaily/wadaily-transcript .
```

### 起動
```bash
docker run --gpus all -p 8000:8000 wadaily.sakuracr.jp/wadaily/wadaily-transcript
```

### テスト
```bash
pip install websockets

# WAVファイルを送信してテスト（24kHz, mono, 16bit）
python test.py test.wav

# サーバー指定
python test.py test.wav ws://192.168.1.100:8000/
```

WAVファイルの変換:
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 test.wav
```

## API

OpenAI Realtime API互換のWebSocket API。

### エンドポイント

`ws://localhost:8000/`

### 送信イベント
```json
{"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
{"type": "input_audio_buffer.commit"}
{"type": "input_audio_buffer.clear"}
```

### 受信イベント
```json
{"type": "session.created", "session": {...}}
{"type": "input_audio_buffer.speech_started", "item_id": "..."}
{"type": "input_audio_buffer.speech_stopped", "item_id": "..."}
{"type": "conversation.item.input_audio_transcription.completed", "transcript": "こんにちは"}
```

### 音声フォーマット

- サンプルレート: 24000 Hz
- チャンネル: モノラル
- ビット深度: 16bit signed integer (PCM16)
- エンコーディング: Base64
