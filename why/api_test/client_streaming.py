import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100 # Adjust this to match your model.config.sampling_rate

async def stream_audio():
    uri = "ws://localhost:8001/ws/generate"
    
    # Initialize the audio output stream
    # float32 is standard for the bytes we are sending
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    stream.start()

    async with websockets.connect(uri) as websocket:
        # 1. Send the request
        payload = {
            "text": "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।",
            "description": "A male speaker with a clear voice and moderate speed."
        }
        await websocket.send_text(json.dumps(payload))
        print("Request sent, waiting for audio...")

        # 2. Receive and play chunks
        try:
            while True:
                message = await websocket.recv()
                
                if isinstance(message, str) and message == "EOS":
                    print("End of audio stream.")
                    break
                
                # Convert bytes back to numpy array
                audio_chunk = np.frombuffer(message, dtype=np.float32)
                
                # Write to the soundcard
                stream.write(audio_chunk)
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            stream.stop()
            stream.close()

if __name__ == "__main__":
    asyncio.run(stream_audio())