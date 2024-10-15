"""实现一个直线类"""

from typing import Any


class Line:
    
    def __init__(self, point0: tuple=None, point1: tuple=None, k=None, b=None) -> None:        
        
        try:
            self.k = (point1[1] - point0[1]) / (point1[0] - point0[0])
            self.b = point1[1] - self.k * point1[0]
        except (ZeroDivisionError, TypeError):
            try:
                self.k = None
                self.b = point0[0]
            except Exception:
                pass
            
        if k is not None:
            self.k = k
        
        if b is not None:
            self.b = b
            
    def __call__(self, **kwargs) -> Any:
        x = kwargs.get("x")
        y = kwargs.get("y")
        
        if self.k == None:
            return y
            
        if x is not None:
            return self.k * x + self.b
        
        if y is not None:
            return (y - self.b) / self.k
        
    def __str__(self) -> str:
        if self.k is not None:
            if self.k != 0:
                return f"y = {self.k} × x + {self.b}"
            else: return f"y = {self.b}"
        else:
            return f"x = {self.b}"
        
    @property
    def sin(self):
        y = 0
        x = self(y = y)
        x = x + 1
        y = self(x = x)
        z = (x ** 2 + y ** 2) ** .5
        return y / z
    
    @property
    def cos(self):
        y = 0
        x = self(y = y)
        x = x + 1
        y = self(x = x)
        z = (x ** 2 + y ** 2) ** .5
        return 1 / z
    
    @property
    def tan(self):
        return self.k
    
def getVerticalLine(point, line: Line):
    
    if line.k is None:
        return Line(k = 0, b = point[1])
    
    if line.k != 0:
        k = - 1 / line.k
    else: 
        return Line(k = 1, b = 1)
    b = point[1] - k * point[0]
    
    return Line(k = k, b = b)
        
if __name__ == "__main__":
    line = Line(k = 1, b = 0)
    print(line)
    print(line(x = 0))
    print(line(y = 1))
    print(line.tan)
    print(line.sin)
    print(line.cos)
    print(getVerticalLine((0, 0), line))