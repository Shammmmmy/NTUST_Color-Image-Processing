#匯入此作業使用到的模組
import cv2 #OpenCV函式庫，用於圖像處理
import os #檔案和目錄操作
import numpy as np #NumPy函式庫，用於數值計算
import matplotlib.pyplot as plt #Matplotlib函式庫，用於圖像顯示

#將圖像轉換為浮點數格式
def img_float(img):
    float_img = img / 255 #將像素值從0-255轉換為0-1
    float_img = float_img.astype('float32') #轉換為32位元浮點數
    
    return float_img

#對圖像進行二值化處理
def image_threshold(img, thrsdh, maxval):
    height, width = img.shape[:2] #獲取圖像的高度和寬度

    #用雙層迴對每個像素進行二值化
    for i in range(height):
        for j in range(width):
            if img[i][j] > 0:
                img[i][j] = maxval #大於threshold的像素設為最大值
            else:
                img[i][j] = thrsdh #小於threshold的像素設為threshold

    return img

#提取圖像的各種特徵
def img_feature(img):
    float_img = img_float(img) #將圖像轉換為浮點數格式

    #LAB空間
    img_lab = cv2.cvtColor(float_img, cv2.COLOR_BGR2LAB) #轉換到LAB色彩空間
    L = img_lab[:, :, 0] #L通道：亮度
    a = img_lab[:, :, 1] #a通道：紅綠對立通道
    b = img_lab[:, :, 2] #b通道：黃藍對立通道
    L_standard, L_mean = np.std(L), np.mean(L) #計算明度L的標準差和平均值
    a_mean, b_mean = np.mean(a), np.mean(b) #計算a.b通道的均值

    #HSV空間
    img_hsv = cv2.cvtColor(float_img, cv2.COLOR_BGR2HSV) #轉換到HSV色彩空間
    H = img_hsv[:, :, 0] #色相通道
    S = img_hsv[:, :, 1] #飽和度通道
    H_mean, S_mean = np.mean(H), np.mean(S) #計算色相和飽和度平均值

    threshold = 5 #設定門檻值

    #高通濾波
    L_laplacian = cv2.Laplacian(L, -1, ksize=3)  #高通濾波
    L_laplacian = np.abs(L_laplacian)  #取絕對值
    _, L_laplacian_threshold = cv2.threshold(L_laplacian, threshold, 255, cv2.THRESH_BINARY) #二值化
    L_laplacian_threshold_mean = np.mean(L_laplacian_threshold)  # 紋理特徵均值

    #Sobel垂直線檢測
    L_sobel = cv2.Sobel(L, -1, 1, 0, ksize=3)  #Sobel濾波器，X方向（垂直邊緣）
    L_sobel = np.abs(L_sobel)  #取絕對值
    _, L_sobel_threshold = cv2.threshold(L_sobel, threshold, 255, cv2.THRESH_BINARY) #二值化
    L_sobel_threshold_mean = np.mean(L_sobel_threshold)  #垂直邊緣均值

    #返回所有提取的特徵
    return {
        'Contrast': L_standard,
        'Lightness': L_mean,
        'Hue': H_mean,
        'Saturation': S_mean,
        'Red-Green': a_mean,
        'Yellow-Blue': b_mean,
        'Texture': L_laplacian_threshold_mean,
        'Vertical edges': L_sobel_threshold_mean
    }

#對特徵值進行正規化處理，將其值縮放到0-500範圍內
def normalize_features(features_list):
    normalized = {}

    #遍歷每個特徵名稱
    for feature in features_list[0].keys():
        values = np.array([f[feature] for f in features_list]) #提取所有圖片在當前特徵上的值，並轉換為numpy陣列
        min_val, max_val = values.min(), values.max() #計算當前特徵值的最小值和最大值

        #當特徵的最大值與最小值相等時，避免除以零的錯誤
        if min_val == max_val:
            #將所有值設置為範圍的中間值（500/2），以保證正規化結果的一致性
            normalized[feature] = np.full_like(values, 250)
        else:
            #正規化計算公式：將值縮放到0~500範圍
            normalized[feature] = np.clip(((values - min_val) / (max_val - min_val) * 500).astype(int), 0, 500)
    
    return normalized

#在黑色背景圖上繪製二維分析圖
def plot_img_feature(images, features, x_feature, y_feature, output):
    black_img = np.zeros((550, 550, 3), np.uint8) #創建黑色背景

    #對每張圖片進行處理
    for i, img in enumerate(images):
        #計算位置
        x = features[x_feature][i]  #轉換為整數
        y = 500 - features[y_feature][i]   #反轉Y軸

        #將圖縮小至50×50尺寸
        resize_img = cv2.resize(img, (50, 50))

        #防止超出邊界
        x = min(max(0, x), 500)
        y = min(max(0, y), 500)

        #貼上圖片
        black_img[y:y + 50, x:x + 50] = resize_img

    #儲存處理後的結果
    cv2.imwrite(output, black_img)
    
    #顯示圖片
    black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB)
    plt.imshow(black_img)
    plt.xlabel(x_feature, fontsize=12) #設置x軸標籤
    plt.ylabel(y_feature, fontsize=12) #設置y軸標籤
    plt.xticks([])
    plt.yticks([])
    plt.show()

#針對資料夾中每個圖像做依序處理
def img_process(img_dir):  
    images = []
    features_list = []

    #讀取並處理所有圖像
    for img_name in sorted(os.listdir(img_dir)): #遍歷指定目錄中的所有影像檔案，並按名稱排序
        img_path = os.path.join(img_dir, img_name) #將影像檔案的名稱與目錄路徑結合
        img = cv2.imread(img_path) #讀取圖像檔案
        images.append(img) #將讀取的圖像添加到images陣列中
        features_list.append(img_feature(img)) #提取當前圖像的特徵，並將特徵字典添加到features_list陣列中

    #特徵值正規化
    norm_features = normalize_features(features_list)
    
    #生成不同特徵組合的二維分析圖結果
    plot_img_feature(images, norm_features, 'Contrast', 'Lightness', 'Contrast_Lightness.jpg')
    plot_img_feature(images, norm_features, 'Hue', 'Saturation', 'Hue_Saturation.jpg')
    plot_img_feature(images, norm_features, 'Red-Green', 'Yellow-Blue', 'Red-Green_Yellow-Blue.jpg')
    plot_img_feature(images, norm_features, 'Texture', 'Vertical edges', 'Texture_Vertical edges.jpg')

if __name__ == '__main__':
    #目標資料夾
    img_path = './imgbank'
    img_process(img_path)
    