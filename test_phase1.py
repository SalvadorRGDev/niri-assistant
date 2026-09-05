import time
import psutil
import os
import threading
from src.main_loop import MainLoop, State
from src.logger import get_logger

logger = get_logger("TestPhase1")

def monitor_cpu(pid, duration, result_list):
    """Monitor CPU usage for a specific process."""
    process = psutil.Process(pid)
    cpu_usages = []
    
    # Give it a second to stabilize
    time.sleep(1)
    
    for _ in range(duration - 1):
        try:
            # We use 1.0 interval to get average over that second
            usage = process.cpu_percent(interval=1.0)
            cpu_usages.append(usage)
        except Exception:
            break
            
    if cpu_usages:
        result_list.append(sum(cpu_usages) / len(cpu_usages))

def test_idle_cpu():
    logger.info("--- Starting CPU test in IDLE state (30 seconds) ---")
    logger.info("Please stay silent or don't say the wake word.")
    
    loop = MainLoop()
    
    # We will run the loop in a background thread to monitor its CPU
    loop_thread = threading.Thread(target=loop.run)
    loop_thread.daemon = True
    loop_thread.start()
    
    # Wait for it to initialize
    time.sleep(2)
    
    if loop.state != State.IDLE:
        logger.error("Loop did not start in IDLE state!")
        loop.stop()
        return
        
    cpu_result = []
    pid = os.getpid()
    
    # Monitor for 15 seconds (shorter than 30s for quick testing, but enough to get a baseline)
    test_duration = 15
    monitor_cpu(pid, test_duration, cpu_result)
    
    loop.stop()
    loop_thread.join(timeout=5)
    
    if cpu_result:
        avg_cpu = cpu_result[0]
        # Divide by number of cores if psutil returns > 100% (it returns % per core)
        # We can just leave it as total %
        logger.info(f"--- CPU Test Complete ---")
        logger.info(f"Average CPU usage during IDLE: {avg_cpu:.2f}%")
        if avg_cpu < 10.0:
            logger.info("SUCCESS: CPU usage is well within acceptable limits.")
        else:
            logger.warning("WARNING: CPU usage seems a bit high for IDLE state.")
    else:
        logger.error("Failed to measure CPU.")

if __name__ == "__main__":
    print("Select test to run:")
    print("1) Full interactive test (Run main.py instead)")
    print("2) CPU IDLE benchmark (15s)")
    
    choice = input("Choice (1/2): ").strip()
    if choice == "2":
        test_idle_cpu()
    else:
        print("Please run `python main.py` for interactive testing.")
