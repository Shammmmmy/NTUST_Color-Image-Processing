#匯入此作業使用到的模組
import cv2 #OpenCV函式庫，用於圖像處理
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

#Laplacian高通濾波
def laplacian(img):
    float_img = img_float(img) #將圖像轉換為浮點數格式
    img_gray = cv2.cvtColor(float_img, cv2.COLOR_RGB2GRAY) #轉換為灰度圖像

    #自訂八方向 Laplacian 濾鏡
    laplacian_f = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])
    
    #應用拉普拉斯濾波器
    laplacian_img = cv2.filter2D(img_gray, -1, laplacian_f)
    laplacian_img = np.abs(laplacian_img) #取絕對值
    
    laplacian_img = np.clip(laplacian_img * 3, 0, 1) # 強化濾波結果並限制範圍0~1
    
    return laplacian_img

#處理前景和背景的高通濾波結果，生成遮罩
def mask_process(fg_hipass, bg_hipass):
    mask = fg_hipass - bg_hipass #計算前景遮罩

    #應用均值濾波
    kernel_size = 15
    mean_f_kernal = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    mean_f_mask = cv2.filter2D(mask, -1, mean_f_kernal)

    #對遮罩進行二值化處理
    threshold_mask = image_threshold(mean_f_mask, 0, 255) 
    
    #顯示並保存二值化後的遮罩
    plt.imshow(threshold_mask, cmap='gray')
    plt.title('二值化後的前景遮罩')
    plt.axis('off')
    plt.savefig('threshold_mask.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    return threshold_mask

if __name__ == "__main__":
    #讀取前景和背景圖像
    fg_img = './depth of field/5fg.JPG'
    bg_img = './depth of field/5bg.JPG'
    
    #讀取並轉換色彩空間從BGR到RGB
    fg = cv2.cvtColor(cv2.imread(fg_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    bg = cv2.cvtColor(cv2.imread(bg_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    #設置matplotlib支援中文顯示
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    #顯示原始的前景和背景圖像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(fg)
    plt.title('對焦在前景(fg)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(bg)
    plt.title('對焦在背景(bg)')
    plt.axis('off')
    plt.show()
    
    #對前景圖像進行Laplacian高通濾波，顯示儲存結果影像
    fg_hipass = laplacian(fg)
    plt.imshow(fg_hipass, cmap='gray')
    plt.title('fg_hipass')
    plt.axis('off')
    plt.savefig('fg_hipass.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    #對背景圖像進行Laplacian高通濾波，顯示並儲存結果影像
    bg_hipass = laplacian(bg)
    plt.imshow(bg_hipass, cmap='gray')
    plt.title('bg_hipass')
    plt.axis('off')
    plt.savefig('bg_hipass.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    #生成並處理遮罩，轉換為uint8格式
    threshold_mask = mask_process(fg_hipass, bg_hipass).astype('uint8')

    #將遮罩應用到前景與背景
    foreground = cv2.bitwise_and(fg, fg, mask=threshold_mask)
    background = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(threshold_mask))

    #合併前景和背景影像
    combine_img = foreground + background
    
    #顯示並保存最終合併結果
    plt.imshow(combine_img)
    plt.title('景深擴增影像')
    plt.axis('off')
    plt.savefig('景深擴增影像.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()