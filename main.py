import os
import sys
from src.main_loop import MainLoop

def main():
    try:
        loop = MainLoop()
        loop.run()
    except Exception as e:
        print(f"Error starting assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
