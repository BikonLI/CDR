import time
from multiprocessing import Process, Value, Queue
import os
import signal



def long_running_function(轮次):
    for i in range(轮次):
        time.sleep(2)  # 模拟耗时操作
    return "Finished"

def putArgsToQue(*args, que: Queue):
    for arg in args:
        que.put(arg)    # 先入先出
    
    que.put("EOA")      # end of args
    
def getArgs(que: Queue):
    args = []
    for i in range(que.qsize()):
        arg = que.get()
        if isinstance(arg, str):
            if arg == "EOA":
                break
        args.append(arg)
    return tuple(args)
    
def run_with_timeout(func, timeout, *args):
    que = Queue()
    
    process = Process(target=func, args=(*args, que))
    process.start()
    
    process.join(timeout)  # 等待指定的超时时间
    
    if process.is_alive():
        print("Function timed out.")
        process.terminate()  # 强制终止进程
        print(111)
        process.join()  # 等待进程结束
        print(222)
        
    result = que.get(timeout=1)
    print(result)
    
    return result

if __name__ == "__main__":

    result_code = run_with_timeout(long_running_function, 5, 3)  # 设置超时时间为 5 秒
    print("Result code:", result_code)
