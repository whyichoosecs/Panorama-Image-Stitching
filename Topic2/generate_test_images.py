# 运行这个快速脚本来创建棋盘格
import cv2
import numpy as np

# 创建简单的9x6内角点棋盘格
img = np.ones((600, 800, 3), dtype=np.uint8) * 255
square_size = 50

for i in range(6):  # 6行内角点 = 7行方格
    for j in range(9):  # 9列内角点 = 10列方格
        if (i + j) % 2 == 0:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        
        x = j * square_size + 100
        y = i * square_size + 100
        
        cv2.rectangle(img, (x, y), (x+square_size, y+square_size), color, -1)

cv2.imwrite('data/calib_images/chessboard.jpg', img)
print("创建了棋盘格图像: data/calib_images/chessboard.jpg")