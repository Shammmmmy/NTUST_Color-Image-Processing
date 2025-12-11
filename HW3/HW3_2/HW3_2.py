#匯入此作業使用到的模組
import os #用於操作系統功能，這裡用在清除畫面
import cv2 #OpenCV函式庫，用於圖像處理
import types #用於類型判斷
import numpy as np #NumPy函式庫，用於數值計算
import matplotlib.pyplot as plt #Matplotlib函式庫，用於圖像顯示

#初始化 
def initialize():
    #清畫面
    os.system('cls' if os.name == 'nt' else 'clear')

    #清變數
    for var in list(globals().keys()):
        if var not in ["__builtins__", "__name__", "__doc__", "__package__"] and not isinstance(globals()[var], (type(os), types.FunctionType)):
            del globals()[var]
    
    #關閉所有figure
    plt.close('all')

    path = 'HW3 instructions\imageset' #設圖庫路徑
    
    X = [] #X序列
    Xf = [] #Xf序列
    Y = [] #Y序列
    Yf = [] #Yf序列

    #迴圈依序產生1到40的影像編號
    for i in range(1, 41):
        img_x_path = f'{path}\\{i}.jpg' #x影像路徑
        img_x = cv2.cvtColor(cv2.imread(img_x_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) #讀取x影像並轉換為RGB
        X.append(img_x) #將x影像存入X序列
        feature_x = cv2.resize(img_x, (1, 3)) #將x影像縮小至3x1x3特徵擷取
        Xf.append(feature_x) #將x影像特徵存入Xf序列

        img_y_path = f'{path}\\{i + 80}.jpg' #y影像路徑
        img_y = cv2.cvtColor(cv2.imread(img_y_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) #讀取y影像並轉換為RGB
        Y.append(img_y) #將y影像存入Y序列
        feature_y = cv2.resize(img_y, (1, 3)) #將y縮小至3x1x3特徵擷取
        Yf.append(feature_y) #將y影像特徵存入Yf序列

    #轉換成numpy array
    X = np.array(X)
    Xf = np.array(Xf) / 255.0 #正規化
    Y = np.array(Y)
    Yf = np.array(Yf) / 255.0 #正規化
    
    return X, Xf, Y, Yf

#參數訓練
def parameter_training(Xf, Yf):
    w = np.zeros((3, 1, 3)) #初始化權重矩陣

    #學習率lr設定0.2
    lr = 0.2

    #儲存每個zone和channel的Loss
    loss_history = [[[[] for _ in range(3)] for _ in range(1)] for _ in range(3)]  
    
    #迴圈
    for zone in range(3): #高、中、低位置迴圈
        for ch in range(3): #RGB通道迴圈
            Xi = [Xf[i, zone, 0, ch] for i in range(len(Xf))] #從Xf序列讀出參數相應的輸入值Xi
            Yi = [Yf[i, zone, 0, ch] for i in range(len(Xf))] #從Yf序列讀出參數相應的輸入值Yi
            
            #迭代迴圈(相當於回合數epoch)設定為30
            for epoch in range(30):    
                #計算L2Loss和梯度
                L2 = 0
                G = 0
                #計算L2 loss及誤差梯度向量
                for i in range(len(Xi)):
                    L2 += ((Yi[i] - w[zone, 0, ch] * Xi[i]) ** 2)
                    G += ((Yi[i] - w[zone, 0, ch] * Xi[i]) * 2 * Xi[i])
                L2 /= len(Xi)
                G /= -len(Xi)
                
                #權重增量向量
                delta_w = -lr * G

                #更新權重
                w[zone, 0, ch] += delta_w
                
                #儲存Loss
                loss_history[zone][0][ch].append(L2)
    
    return w, loss_history

#展示成果
def result(X, Xf, Y, Yf, w):
    w_img = np.zeros((3, 3, 3)) #創建3x3的影像，3個區域，3個顏色通道
    for ch in range(3):
        w_img[ch] = w[ch, 0] #將每個區域的RGB權重填入影像中

    #將3x3的權重影像使用線性內差方法放大到224x224
    w_resize = cv2.resize(w_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    #從0-39中隨機選擇前4個不重複的數字
    rand = np.random.permutation(40)[:4]

    #建立一個大的畫布，用於顯示所有影像
    plt.figure(figsize=(24, 16))

    #依序處理選中的影像索引
    for index, value in enumerate(rand):
        original_img = X[value] #原始影圖
        style_img = Y[value] #對應的風格影像

        #建立一個與原始影像相同大小的畫布，用於存放轉換後的影像，dtype設為float32以支持浮點數運算
        transform_img = np.ones_like(original_img, dtype=np.float32)

        #對影像的每個像素進行處理
        for i in range(224):
            for j in range(224):
                for c in range(3):
                    #將原始影像的像素值與對應位置的權重相乘
                    transform_img[i, j, c] = original_img[i, j, c] * w_resize[i, j, c]

        #將像素值限制在0-255範圍內，並轉換為整數類型
        transform_img = np.clip(transform_img, 0, 255).astype(np.uint8)
        
        #顯示原始影像
        plt.subplot(3, 4, index + 1)
        plt.imshow(original_img)
        plt.axis('off') #關閉座標軸
        plt.title(f'content X ({value + 1})')

        #顯示目標風格影像
        plt.subplot(3, 4, index + 5)
        plt.imshow(style_img)
        plt.axis('off') #關閉座標軸
        plt.title(f'style Y ({value + 81})')

        #顯示風格轉換後的影像
        plt.subplot(3, 4, index + 9)
        plt.imshow(transform_img)
        plt.axis('off') #關閉座標軸
        plt.title(f'X ({value + 1}) to Y')

    # 顯示圖像
    plt.tight_layout() #自動調整圖像間距
    plt.get_current_fig_manager().window.state('zoomed') #讓圖形全螢幕顯示
    plt.savefig('result.jpg')  #儲存圖片
    plt.show()

#繪製9個參數的 Loss 梯度下降歷史曲線
def plot_loss(loss_history):
    #繪製Loss曲線在同一個圖中
    fig, ax = plt.subplots(3, 3, figsize=(12, 12)) 

    #處理每個位置和通道，繪製對應的Loss曲線
    for zone in range(3): #高、中、低位置迴圈
        for ch in range(3): #RGB通道迴圈
            axs = ax[zone, ch] #當前子圖的axes對象
            axs.plot(loss_history[zone][0][ch]) #繪製當前位置和通道的Loss曲線
            axs.set_title(f"zone {zone+1}, channel {['1', '2', '3'][ch]}") #設置標題
            axs.set_xlabel('epoch') #x軸標籤
            axs.set_ylabel('loss') #y軸標籤

    plt.tight_layout(rect=[0, 0, 1, 0.95]) #調整整體布局
    plt.get_current_fig_manager().window.state('zoomed') #讓圖形全螢幕顯示
    plt.savefig('loss history.jpg') #儲存圖片
    plt.show()

if __name__ == '__main__':
    #初始化資料，載入影像資料並提取特徵
    X, Xf, Y, Yf = initialize()

    #執行參數訓練，使用Xf和Yf中的特徵來訓練，並返回最終的權重w和loss歷史
    w, loss_history = parameter_training(Xf, Yf)

    #將原影像、目標風格影像、以及轉換後的影像隨機展示
    result(X, Xf, Y, Yf, w)

    #繪製loss歷史的曲線圖
    plot_loss(loss_history)
    