#匯入此作業使用到的模組
import cv2 #OpenCV函式庫，用於圖像處理
import numpy as np #NumPy函式庫，用於數值計算
import matplotlib.pyplot as plt #Matplotlib函式庫，用於圖像顯示

#將圖像轉換為浮點數格式
def img_float(img):
    float_img = img / 255 #將像素值從0-255轉換為0-1
    float_img = float_img.astype('float32') #轉換為32位元浮點數
    
    return float_img

#模擬紅綠色盲的視覺效果
def red_green_colorblindness(image):
    float_img = img_float(image) #將圖像轉換為浮點數格式
    float_img_lab = cv2.cvtColor(float_img, cv2.COLOR_RGB2LAB) #將RGB色彩空間轉換為LAB色彩空間

    #分別得到LAB三個空間
    L = float_img_lab[:, :, 0] #L通道：亮度
    a = float_img_lab[:, :, 1] #a通道：紅綠對立通道
    b = float_img_lab[:, :, 2] #b通道：黃藍對立通道
    
    #將a通道設為0
    a[:] = 0
    
    #合併修改後的LAB空間
    lab_img_modify = np.zeros_like(float_img_lab)
    lab_img_modify[:, :, 0] = L
    lab_img_modify[:, :, 1] = a
    lab_img_modify[:, :, 2] = b

    #將LAB色彩空間轉回RGB
    rgb_img = cv2.cvtColor(lab_img_modify, cv2.COLOR_LAB2RGB)

    #顯示並儲存結果
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title('紅綠色盲')
    plt.savefig('red_green_colorblindness.jpg', bbox_inches='tight', pad_inches=0) #儲存圖片並將空白去除
    plt.show()

#模擬黃藍色盲的視覺效果
def yellow_blue_colorblindness(img):
    float_img = img_float(img) #將圖像轉換為浮點數格式
    float_img_lab = cv2.cvtColor(float_img, cv2.COLOR_RGB2LAB) #將RGB色彩空間轉換為LAB色彩空間

    #分別得到LAB三個空間
    L = float_img_lab[:, :, 0] #L通道：亮度
    a = float_img_lab[:, :, 1] #a通道：紅綠對立通道
    b = float_img_lab[:, :, 2] #b通道：黃藍對立通道

    #將b通道設為0
    b[:] = 0

    #合併修改後的LAB空間
    lab_img_modify = np.zeros_like(float_img)
    lab_img_modify[:, :, 0] = L
    lab_img_modify[:, :, 1] = a
    lab_img_modify[:, :, 2] = b

    #將LAB色彩空間轉回RGB
    rgb_img = cv2.cvtColor(lab_img_modify, cv2.COLOR_LAB2RGB)

    #顯示並儲存結果
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title('黃藍色盲')
    plt.savefig('yellow_blue_colorblindness.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

#模擬青光眼的視覺效果
def glaucoma(img):
    height, width = img.shape[0:2] #獲取圖像的高度和寬度
    sigma = 100 #設定sigma
    
    #建立與影像同尺寸的2D高斯濾鏡
    gaussian = np.outer(cv2.getGaussianKernel(height, sigma), (cv2.getGaussianKernel(width, sigma)).T)
    gaussian /= np.max(gaussian) #將濾鏡數值矩陣的每個數值除以其最大值

    # 對圖像應用高斯濾波
    gaussian_img = img.astype('float32')
    for color in range(3): #對每個色彩通道分別處理
        gaussian_img[:, :, color] *= gaussian

    #將結果裁減回0~255範圍，並轉換回uint8格式
    gaussian_img = np.clip(gaussian_img, 0, 255).astype('uint8')

    #顯示並儲存結果
    plt.imshow(gaussian_img)
    plt.axis('off')
    plt.title('青光眼')
    plt.savefig('glaucoma.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    #讀取原始圖像並轉換色彩空間從BGR到RGB
    original_img = cv2.imread('picture.jpg')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    #設置matplotlib支援中文顯示
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    #顯示原始圖像
    plt.imshow(original_img)
    plt.axis('off') #關閉坐標軸
    plt.title('一般人')
    plt.show()

    red_green_colorblindness(original_img) #模擬紅綠色盲
    yellow_blue_colorblindness(original_img) #模擬黃藍色盲
    glaucoma(original_img) #模擬青光眼
    
    