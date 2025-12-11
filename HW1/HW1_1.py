#匯入此作業使用到的模組
import cv2  #用於讀取附件灰階影像與儲存影像
import matplotlib.pyplot as plt #用於顯示影像
import random #用於產生亂數
import numpy as np #建立和影像相同大小的空白新影像矩陣與計算高斯分布所用x雜訊值的範圍
import math #使用e.pi常數與開根號

#定義加入高斯雜訊函式，img為影像，sigma為控制高斯分布離散程度的標準差
def add_gaussian_noise(img, sigma):
    #使用變數取得影像的高與寬
    height = img.shape[0]
    width = img.shape[1]

    #建立和影像相同大小的空白新影像矩陣
    gaussian = np.array([[0 for j in range(width)] for i in range(height)])
    
    #利用公式計算高斯分布，建立從-100到100的範圍x值
    x = np.arange(-100, 101)
    exponent = -1 * ((x ** 2) / (2 * (sigma ** 2))) #高斯分布指數
    y = (1 / (sigma * math.sqrt(2 * math.pi))) * (math.e ** exponent) #計算高斯分布y值
    
    #計算機率直方圖
    sum_y = 0
    for i in y:
        sum_y += i #計算y的總和
    y = y / sum_y #正規化y值
    
    #累積直方圖
    cumsum_sum = 0
    len_y = 0
    for i in y:
        len_y += 1 #計算y的長度
    y_cum = [0] * len_y #建立一個一維陣列存放累計機率
    for i in range(len_y): #利用y長度累加機率值
        cumsum_sum += y[i] #累加當前機率值
        y_cum[i] = cumsum_sum #將累積的值存入陣列

    #用雙層迴圈添加高斯雜訊
    for i in range(height):
        for j in range(width):
            rand = random.uniform(0, 1) #產生0~1之間的隨機數
            for k in range(len_y): #搜尋累積高斯曲線對應的雜訊數值
                if y_cum[k] > rand:
                    x_noise = x[k]  #選擇對應的x雜訊值
                    break
            gaussian[i][j] = img[i][j] #保留原影像像素值
            gaussian[i][j] += x_noise   #添加雜訊到像素值
            
    return gaussian #返回加入高斯雜訊的影像

#定義加入胡椒鹽雜訊函式，img為影像，sap_ratio為胡椒鹽雜訊比例
def add_saltandpepper_noise(img, sap_ratio):
    #使用變數取得影像的高與寬
    height = img.shape[0]
    width = img.shape[1]

    #建立和影像相同大小的空白新影像矩陣
    saltandpepper = np.array([[0 for j in range(width)] for i in range(height)])

    #用雙層迴圈添加胡椒鹽雜訊
    for i in range(height):
        for j in range(width):
            rand = random.uniform(0, 100) #產生0~100的隨機數
            if rand > sap_ratio: #若隨機數大於雜訊比例，則保留原影像像素值
                saltandpepper[i][j] = img[i][j]
            else: #若隨機數小於雜訊比例，則隨機決定黑或白點
                if (random.uniform(0, 1) < 0.5): #產生0~1隨機數判斷
                    saltandpepper[i][j] = 255 #若隨機數<0.5則設為白點(鹽)
                else:
                    saltandpepper[i][j] = 0 #若隨機數>0.5則設為黑點(胡椒)
    
    return saltandpepper #返回加入胡椒鹽雜訊的影像

if __name__  == "__main__":
    #讀取附件的 8-bit 灰階影像
    img = 'ntust_gray.jpg' 
    original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) #以灰階讀取影像

    #顯示輸入影像
    plt.imshow(original_img, cmap='gray') #用灰階模式顯示影像
    plt.title('original img')
    plt.show()

    #輸入雜訊類型選項
    option = int(input("輸入雜訊類型選項 1.高斯雜訊 2.胡椒鹽雜訊 : "))

    #根據選項輸入雜訊的參數：高斯分配的標準差，或胡椒鹽雜訊的濃度
    if option == 1:
        sigma = int(input("輸入高斯分配的標準差參數 : "))
        gaussian_noise_img = add_gaussian_noise(original_img, sigma) #加入高斯雜訊
        cv2.imwrite('add Gaussian.jpg', gaussian_noise_img) #儲存結果影像
        plt.imshow(gaussian_noise_img, cmap='gray') #顯示加入雜訊後的影像
        plt.title('add gaussian')
        plt.show()
    elif option == 2:
        sap_ratio = int(input("輸入胡椒鹽雜訊的濃度參數 : "))
        saltandpepper_noise_img = add_saltandpepper_noise(original_img, sap_ratio) #加入胡椒鹽雜訊
        cv2.imwrite('add Salt and Pepper.jpg', saltandpepper_noise_img) #儲存結果影像
        plt.imshow(saltandpepper_noise_img, cmap='gray') #顯示加入雜訊後的影像
        plt.title('add Salt and Pepper')
        plt.show()
