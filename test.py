import asyncio
import websockets
import json
import base64
import wave
import sys


async def test_with_wav(wav_path: str, server_url: str = "ws://localhost:8000/"):
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    print(f"WAV: {wav_path}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Channels: {channels}")
    print(f"  Sample Width: {sample_width} bytes")
    print(f"  Duration: {len(frames) / (sample_rate * channels * sample_width):.2f}s")

    if sample_rate != 24000:
        print(f"Warning: Sample rate is {sample_rate}, expected 24000")
    if channels != 1:
        print(f"Warning: Channels is {channels}, expected 1 (mono)")
    if sample_width != 2:
        print(f"Warning: Sample width is {sample_width}, expected 2 (16bit)")

    async with websockets.connect(server_url) as ws:
        session = await ws.recv()
        print(f"\n← {json.loads(session)['type']}")

        async def recv_events():
            async for msg in ws:
                event = json.loads(msg)
                event_type = event["type"]
                if "transcript" in event:
                    print(f"← {event_type}: {event['transcript']}")
                else:
                    print(f"← {event_type}")

        recv_task = asyncio.create_task(recv_events())

        chunk_size = int(24000 * 0.1) * 2  # 100ms分

        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            audio_b64 = base64.b64encode(chunk).decode()

            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }))
            await asyncio.sleep(0.1)

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("\n→ input_audio_buffer.commit")

        await asyncio.sleep(3)
        recv_task.cancel()


if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/"
    print(wav_path, url)

    asyncio.run(test_with_wav(wav_path, url))
