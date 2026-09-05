import time
import numpy as np
from src.audio import AudioStream
from src.vad import VoiceActivityDetector

def main():
    audio = AudioStream()
    vad = VoiceActivityDetector()
    
    print("Iniciando micrófono para probar VAD. Habla al micrófono...")
    audio.start()
    
    try:
        while True:
            chunk = audio.read_chunk(512)
            prob = vad.process_chunk(chunk)
            
            # Crear una barra visual
            bars = int(prob * 50)
            bar_str = "#" * bars + "-" * (50 - bars)
            print(f"VAD Prob: {prob:.3f} | [{bar_str}]", end="\r")
            
    except KeyboardInterrupt:
        print("\nPrueba terminada.")
    finally:
        audio.stop()

if __name__ == "__main__":
    main()
