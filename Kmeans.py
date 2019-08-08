def Cluster(lines, var1, var2):
    '''
    数轴上两类聚类
    输入：初始line，两个估计聚集点
    输出：两个聚类后的数组
    '''
    while True:
        line1 = []
        line2 = []
        for line in lines:
            for rho, _ in line:
                if abs(abs(rho) - var1) <= abs(abs(rho) - var2):
                    line1.append(line)
                else:
                    line2.append(line)
        nvar1 = getMeans(line1)
        nvar2 = getMeans(line2)
        # 若新聚集点误差大于原聚集点的百分之一则继续
        if abs(nvar1 - var1) > var1 / 100 or abs(nvar2 - var2) > var2 / 100:
            var1 = nvar1
            var2 = nvar2
        else:
            return line1, line2

def getMeans(lines):
    '''
    计算平均值，line为数组
    '''
    sum = 0
    for line in lines:
        for rho, _ in line:
            sum += abs(rho)
    return (sum / len(lines)) if len(lines) else 0