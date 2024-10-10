import threading
import multiprocessing


class MultiTask:
    def __init__(self, tasksNum: int, totalWork: list=[]) -> None:
        self.tasks = []
        self.totalWork = totalWork
        self.tasksNum = tasksNum
        self.lock = multiprocessing.Lock()
        pass
    
    def mallocWork(self, totalWork):
        """分配任务，需要复写

        Args:
            totalWork (list[work0, work1, ...]): 任务列表

        Returns:
            list[list[work0, work1, ...], list[workn, ...]]: 二维任务列表
        """
        return None
    
    def createTask(self, work):
        """需要复写

        Args:
            work (list): 一个任务

        Returns:
            Process: 返回一个任务句柄
        """
        return None
    
    def createTasks(self):
        works = self.mallocWork(self.totalWork)
        for i in range(self.tasksNum):
            self.tasks.append(self.createTask(works[i]))
            
        return self
    
    def start(self):
        self.createTasks()
        for task in self.tasks:
            task.start()
            
        for task in self.tasks:
            task.join()


# --- 
totalWork = [f"task{i}" for i in range(9)]
import time
def processFunction(work: list):        # 工作函数需要定义在全局作用域
    for w in work:
        for i in range(10000):
            time.sleep(0.0001)
        print(w)

if __name__ == "__main__":
    # 例子
    
    class MyMultiTask(MultiTask):
        
        def mallocWork(self, totalWork):    
            return totalWork[:3], totalWork[3:6], totalWork[6:]
        
        def createTask(self, work):
            p = multiprocessing.Process(target=processFunction, args=(work, ))
            return p
        
    mmt = MyMultiTask(3, totalWork=totalWork)
    mmt.start()