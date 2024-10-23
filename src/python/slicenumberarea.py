from line import *


def findMiddlePoint(point0, point1, weight: float=.5):
    """传入一个系数，返回数字中心点坐标

    Args:
        point0 (tuple): 点
        
        point1 (tuple): 点
        
        weight (float): 系数
        

    Returns:
        tuple[int, int]: 中心点
    """
    x0, x1 = sorted((point0[0], point1[0]))
    x = x0 + (x1 - x0) * (1 - weight)
    y0, y1 = sorted((point0[1], point1[1]))
    y = y0 + (y1 - y0) * (1 - weight)

    return int(x), int(y)


def genRectangle(lsholder, rsholder, rhip, middle):
    """生成一个矩形

    Args:
        point (tuple[int, int]): 中心点
        line0 (Line): 水平线
        line1 (Line): 垂直线
        line2 (Line): 肩膀线
        point0 (tuple[x, y]): 左肩
        point1 (tuple[x, y]): 右肩
    """
    lx, rx = sorted((lsholder[0], rsholder[0]))
    
    x_ratio = .1            # 框的左右界相对于肩膀宽度的比例。越大，框越宽。为0则与肩同宽。
    x0 = lx - (rx - lx) * x_ratio
    x1 = rx + (rx - lx) * x_ratio
    biasT = abs(lsholder[1] - middle[1])
    biasB = abs(rhip[1] - middle[1])
    
    top_ratio = .9          # 框上界相对于肩膀的比例，越大框上界越高
    bot_ratio = .8         # 框下届对于髋的比例，越大框下界越往下
    y0 = middle[1] - biasT * top_ratio         # 框上界的位置。
    y1 = middle[1] + biasB * bot_ratio
    
    dx = rsholder[0] - lsholder[0]
    dy = rsholder[1] - lsholder[1]
    
    biasRatio = .2     # 左右偏置比例，会改变框的位置，不改变框的大小。越大偏的越多。
    if dx * dy > 0:
        xBias = -(x1 - x0) * biasRatio
    else:
        xBias = (x1 - x0) * biasRatio
        
    x0 += xBias
    x1 += xBias
    
    xscale = .90     # 越小框越小，不会改变框位置
    yscale = .90
    
    x0 += (x1 - x0) * (1 - xscale) / 2
    x1 -= (x1 - x0) * (1 - xscale) / 2
    
    y0 += (y1 - y0) * (1 - yscale) / 2
    y1 -= (y1 - y0) * (1 - yscale) / 2

    return (int(x0), int(y0)), (int(x1), int(y1))



def sliceNumberArea(points: tuple, weight: float=.65, weight2: float=.95, weight3: float=.3):
          
    rsholder, lsholder, rhip, lhip, mt, md = points
    
    if (rsholder[0] - lsholder[0]) < 0 and (rhip[0] - lhip[0]) < 0:
        return None
    
    middlePoint = findMiddlePoint(mt, md, weight)

    point1, point2 = genRectangle(lsholder, rsholder,  rhip, middlePoint)
    
    return point1, point2
