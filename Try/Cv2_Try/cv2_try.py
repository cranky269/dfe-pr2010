import cv2
 
img = cv2.imread(r'C:\Users\lenovo\Desktop\ASP\Distance_Question\dfe-pr2010\input1.png', 1)
cv2.imshow('cat', img)   #第一个参数为图片名称，第二个参数为我们输入的矩阵
cv2.waitKey(0)  #0为不限时间,接收到键盘指令才会继续。也可以输入正整数，单位为毫秒。
# 例如 cv2.waitKey(5000) 5000毫秒后会运行cv2.destroyAllWindows() 在5000毫秒期间你仍然可以# 通过任意键来继续
cv2.destroyAllWindows()  #关闭我们所有的窗口