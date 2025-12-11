#匯入此作業使用到的模組
import cv2 #用於讀取附件灰階影像與儲存影像
import matplotlib.pyplot as plt #用於顯示影像
import numpy as np #建立和影像相同大小的空白新影像矩陣

#定義均值濾波函式，img為影像，對影像每個像素應用3X3均值濾波
def mean_filter(img):
    #使用變數取得影像的高與寬
    height = img.shape[0]
    width = img.shape[1]

    #建立和影像相同大小的空白新影像矩陣
    new_image = np.array([[0 for j in range(width)] for i in range(height)])

    #用雙層迴圈依序對(x,y)像素為中心的3×3區域，點對點乘上數值皆為1/9的3×3濾鏡，將加總的數值填入輸出影像的(x,y)位置
    for i in range(height):
        for j in range(width):
            filter = [[0 for j in range(3)] for i in range(3)] #建立3x3的濾波器
            #計算3x3範圍內的均值
            for ni in range(-1, 2):
                for nj in range(-1, 2):
                    if 0 <= i + ni < height and 0 <= j + nj < width: #檢查範圍是否超過影像
                        filter[1 + ni][1 + nj] = img[i + ni][j + nj] #若沒有超過範圍則加入影像像素值
                    else:
                        filter[1 + ni][1 + nj] = 0 #若超過範圍設成0
            mean_value = 0
            #計算3x3區域內像素值的總和
            for k in range(3):
                for v in range(3):
                    mean_value += filter[k][v]
            new_image[i][j] = mean_value * (1 / 9)  #計算均值並填入新影像像素值

    return new_image #返回經過均值濾波後的新影像

#定義中值濾波函式，img為影像，對影像每個像素應用3x3中值濾波
def median_filter(img):
    #使用變數取得影像的高與寬
    height = img.shape[0]
    width = img.shape[1]

    #建立和影像相同大小的空白新影像矩陣
    new_image = np.array([[0 for j in range(width)] for i in range(height)])

    #用雙層迴圈，依序對以(x,y)像素為中心的3×3區域，做灰階值排序。將排序後的中位數，填入輸出影像的(x,y)位置
    for i in range(height):
        for j in range(width):
            filter = [[0 for j in range(3)] for i in range(3)] #建立3x3的濾波器
            #取得3x3範圍內的像素值
            for ni in range(-1, 2):
                for nj in range(-1, 2):
                    if 0 <= i + ni < height and 0 <= j + nj < width: #檢查範圍是否超過影像
                        filter[1 + ni][1 + nj] = img[i + ni][j + nj] #若沒有超過範圍則加入影像像素值
                    else:
                        filter[1 + ni][1 + nj] = 0 #若超過範圍設成0

            # 將3x3範圍內的值轉成一維陣列
            filter_1D = [filter[k][v] for v in range(3) for k in range(3)]
            
            #進行灰階值排序
            for k in range(9):
                for v in range(0, 9 - k - 1):
                    if filter_1D[v + 1] < filter_1D[v]: #如果第v項大於v+1項則交換
                        temp = filter_1D[v + 1]
                        filter_1D[v + 1] = filter_1D[v]
                        filter_1D[v] = temp
            median_value = filter_1D[4] #取得排序後的中位數
            new_image[i][j] = median_value #將中位數值填入新影像像素值

    return  new_image #返回經過中值濾波後的新影像


if __name__ == "__main__":
    #輸入空間濾波選項
    option = int(input("輸入空間濾鏡選項 1.均值濾波 2.中值濾波 : "))

    if option == 1:
        #讀取並顯示含有高斯雜訊的影像
        img1 = 'add Gaussian.jpg'
        gaussian_noise_img = cv2.imread(img1, cv2.IMREAD_GRAYSCALE) #以灰階讀取影像
        plt.imshow(gaussian_noise_img, cmap='gray') #用灰階模式顯示影像
        plt.title('gaussian noise img')
        plt.show()
        #對影像應用均值濾波並顯示結果
        gaussian_noise_mean_filter_img = mean_filter(gaussian_noise_img)
        cv2.imwrite('mean filter img.jpg', gaussian_noise_mean_filter_img) #儲存結果影像
        plt.imshow(gaussian_noise_mean_filter_img, cmap='gray') #顯示經過均值濾波後的影像
        plt.title('gaussian noise meanfilter')
        plt.show()
    elif option == 2:
        #讀取並顯示含有胡椒鹽雜訊的影像
        img2 = 'add Salt and Pepper.jpg'
        saltandpepper_noise_img = cv2.imread(img2, cv2.IMREAD_GRAYSCALE) #以灰階讀取影像
        plt.imshow(saltandpepper_noise_img, cmap='gray') #用灰階模式顯示影像
        plt.title('saltandpepper noise img')
        plt.show()
        #對影像應用中值濾波並顯示結果
        saltandpepper_noise_median_filter_img = median_filter(saltandpepper_noise_img)
        cv2.imwrite('median filter img.jpg', saltandpepper_noise_median_filter_img) #儲存結果影像
        plt.imshow(saltandpepper_noise_median_filter_img, cmap='gray') #顯示經過中值濾波後的影像
        plt.title('saltandpepper noise meanfilter')
        plt.show()