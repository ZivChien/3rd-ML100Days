![roadmap](https://ai100-3.cupoy.com/images/learnWithCoach.png)

* [排名賽](https://ai100-3.cupoy.com/ranking/homeworkrank) top 5%
* [個人主頁](https://ai100-3.cupoy.com/participator/84F13837/questions) 

## 1. 機器學習概論
> 從概念上理解機器學習的目的與限制，並導覽機器學習流程
1. [資料介紹與評估資料 (申論+程式碼)](homework/Day_001_HW.ipynb)
`挑戰是什麼?動手分析前請三思`
2. [機器學習概論 (申論題)](homework/Day_002_HW.ipynb)
`機器學習、深度學習與人工智慧差別是甚麼? 機器學習又有甚麼主題應用?`
3. [機器學習 - 流程與步驟 (申論題)](homework/Day_003_HW.ipynb)
`資料前處理 > 訓練/測試集切分 >選定目標與評估基準 > 建立模型 > 調整參數。熟悉整個 ML 的流程`
4. [EDA/讀取資料與分析流程](homework/Day_004_HW.ipynb)
`如何讀取資料以及萃取出想要了解的信息`

## 2. 資料清理數據前處理
> 以滾動方式進行資料清理與探索性分析
5. [如何新建一個 dataframe?](homework/Day_005-1_HW.ipynb) [如何讀取其他資料?](homework/Day_005-2_HW.ipynb) (非csv的資料)
`1. 從頭建立一個 dataframe 2. 如何讀取不同形式的資料 (如圖檔、純文字檔、json 等)`
6. [EDA: 欄位的資料類型介紹及處理](homework/Day_006_HW.ipynb)
`了解資料在 pandas 中可以表示的類型`
7. [特徵類型](homework/Day_007_HW.ipynb)
`特徵工程依照特徵類型，做法不同，大致可分為數值/類別/時間型三類特徵`
8. [EDA資料分佈](homework/Day_008_HW.ipynb)
`用統計方式描述資料`
9. [EDA: Outlier 及處理](homework/Day_009_HW.ipynb)
`偵測與處理例外數值點：1. 透過常用的偵測方法找到例外 2. 判斷例外是否正常 (推測可能的發生原因)`
10. [數值型特徵 - 去除離群值](homework/Day_010_HW.ipynb)
`數值型特徵若出現少量的離群值，則需要去除以保持其餘數據不被影響`
11. [常用的數值取代：中位數與分位數連續數值標準化](homework/Day_011_HW.ipynb)
`偵測與處理例外數值 1. 缺值或例外取代 2. 數據標準化`
12. [數值型特徵 - 補缺失值與標準化](homework/Day_012_HW.ipynb)
`數值型特徵首先必須填補缺值與標準化，在此複習並展示對預測結果的差異`
13. [DataFrame operationData frame merge/常用的 DataFrame](homework/Day_013_HW.ipynb)
`1. 常見的資料操作方法 2. 資料表串接`
14. [程式實作 EDA: correlation/相關係數簡介](homework/Day_014_HW.ipynb)
`1. 了解相關係數 2. 利用相關係數直觀地理解對欄位與預測目標之間的關係`
15. [DA from Correlation](homework/Day_015_HW.ipynb)
`深入了解資料，從 correlation 的結果下手`
16. [EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)](homework/Day_016_HW.ipynb)
`1. 如何調整視覺化方式檢視數值範圍 2. 美圖修修 - 轉換繪圖樣式`
17. [EDA: 把連續型變數離散化](homework/Day_017_HW.ipynb)
`簡化連續性變數`
18. [程式實作 把連續型變數離散化](homework/Day_018_HW.ipynb)
`深入了解資料，從簡化後的離散變數下手`
19. [Subplots](homework/Day_019_HW.ipynb)
`探索性資料分析 - 資料視覺化 - 多圖檢視 1. 將數據分組一次呈現 2. 把同一組資料相關的數據一次攤在面前`
20. [Heatmap & Grid-plot](homework/Day_020_HW.ipynb)
`探索性資料分析 - 資料視覺化 - 熱像圖 / 格狀圖 1. 熱圖：以直觀的方式檢視變數間的相關性 2. 格圖：繪製變數間的散佈圖及分布`
21. [模型初體驗 Logistic Regression](homework/Day_021_HW.png)
`在我們開始使用任何複雜的模型之前，有一個最簡單的模型當作 baseline 是一個好習慣`

## 3. 資料科學特徵工程技術
> 使用統計或領域知識，以各種組合調整方式，生成新特徵以提升模型預測力。
22. [特徵工程簡介](homework/Day_022_HW.ipynb)
`介紹機器學習完整步驟中，特徵工程的位置以及流程架構`
23. [數值型特徵 - 去除偏態](homework/Day_023_HW.ipynb)
`數值型特徵若分布明顯偏一邊，則需去除偏態以消除預測的偏差`
24. [類別型特徵 - 基礎處理](homework/Day_024_HW.ipynb)
`介紹類別型特徵最基礎的作法 : 標籤編碼與獨熱編碼`
25. [類別型特徵 - 均值編碼](homework/Day_025_HW.ipynb)
`類別型特徵最重要的編碼 : 均值編碼，將標籤以目標均值取代`
26. [類別型特徵 - 其他進階處理](homework/Day_026_HW.ipynb)
`類別型特徵的其他常見編碼 : 計數編碼對應出現頻率相關的特徵，雜湊編碼對應眾多類別而無法排序的特徵`
27. [時間型特徵](homework/Day_027_HW.ipynb)
`時間型特徵可抽取出多個子特徵，或周期化，或取出連續時段內的次數`
28. [特徵組合 - 數值與數值組合](homework/Day_028_HW.ipynb)
`特徵組合的基礎 : 以四則運算的各種方式，組合成更具預測力的特徵`
29. [特徵組合 - 類別與數值組合](homework/Day_029_HW.ipynb)
`類別型對數值型特徵可以做群聚編碼，與目標均值編碼類似，但用途不同`
30. [特徵選擇](homework/Day_030_HW.ipynb)
`介紹常見的幾種特徵篩選方式`
31. [特徵評估](homework/Day_031_HW.ipynb)
`介紹並比較兩種重要的特徵評估方式，協助檢測特徵的重要性`
32. [分類型特徵優化 - 葉編碼](homework/Day_032_HW.ipynb)
`葉編碼 : 適用於分類問題的樹狀預估模型改良`

## 4. 機器學習基礎模型建立
> 學習透過Scikit-learn等套件，建立機器學習模型並進行訓練！
33. [機器如何學習?](homework/Day_033_HW.ipynb)
`了解機器學習的定義，過擬合 (Overfit) 是甚麼，該如何解決`
34. [訓練/測試集切分的概念](homework/Day_034_HW.ipynb)
`為何要做訓練/測試集切分？有什麼切分的方法？`
35. [regression vs. classification](homework/Day_035_HW.ipynb)
`回歸問題與分類問題的區別？如何定義專案的目標`
36. [評估指標選定/evaluation metrics](homework/Day_036_HW.ipynb)
`專案該如何選擇評估指標？常用指標有哪些？`
37. [regression model 介紹 - 線性迴歸/羅吉斯回歸](homework/Day_037_HW.ipynb)
`線性迴歸/羅吉斯回歸模型的理論基礎與使用時的注意事項`
38. [regression model 程式碼撰寫](homework/Day_038_HW.ipynb)
`如何使用 Scikit-learn 撰寫線性迴歸/羅吉斯回歸模型的程式碼`
39. [regression model 介紹 - LASSO 回歸/ Ridge 回歸](homework/Day_039_HW.ipynb)
`LASSO 回歸/ Ridge 回歸的理論基礎與與使用時的注意事項`
40. [regression model 程式碼撰寫](homework/Day_040_HW.ipynb)
`使用 Scikit-learn 撰寫 LASSO 回歸/ Ridge 回歸模型的程式碼`
41. [tree based model - 決策樹 (Decision Tree) 模型介紹](homework/Day_041_HW.ipynb)
`決策樹 (Decision Tree) 模型的理論基礎與使用時的注意事項`
42. [tree based model - 決策樹程式碼撰寫](homework/Day_042_HW.ipynb)
`使用 Scikit-learn 撰寫決策樹 (Decision Tree) 模型的程式碼`
43. [tree based model - 隨機森林 (Random Forest) 介紹](homework/Day_043_HW.ipynb)
`隨機森林 (Random Forest)模型的理論基礎與使用時的注意事項`
44. [tree based model - 隨機森林程式碼撰寫](homework/Day_044_HW.ipynb)
`使用 Scikit-learn 撰寫隨機森林 (Random Forest) 模型的程式碼`
45. tree based model - 梯度提升機 (Gradient Boosting Machine) 介紹
`梯度提升機 (Gradient Boosting Machine) 模型的理論基礎與使用時的注意事項`
46. [tree based model - 梯度提升機程式碼撰寫](homework/Day_046_HW.ipynb)
`使用 Scikit-learn 撰寫梯度提升機 (Gradient Boosting Machine) 模型的程式碼`
## 5. 機器學習調整參數
> 了解模型內的參數意義，學習如何根據模型訓練情形來調整參數
47. [超參數調整與優化](homework/Day_047_HW.ipynb)
`什麼是超參數 (Hyper-paramter) ? 如何正確的調整超參數？常用的調參方法為何？`
48. [Kaggle 競賽平台介紹](homework/Day_048_HW.png)
`介紹全球最大的資料科學競賽網站。如何參加競賽？`
49. [集成方法 : 混合泛化(Blending)](homework/Day_049_HW.png)
`什麼是集成? 集成方法有哪些? Blending 的寫作方法與效果為何?`
50. [集成方法 : 堆疊泛化(Stacking)](homework/Day_050_HW.png)
`Stacking 的設計方向與主要用途是什麼? 通常會使用什麼套件實作?`
## 6. 非監督式機器學習
> 利用分群與降維方法探索資料模式
54. [clustering 1 非監督式機器學習簡介](homework/Day_054_HW.ipynb)
`非監督式學習簡介、應用場景`
55. [clustering 2 聚類算法](homework/Day_055_HW.ipynb)
`K-means`
56. [K-mean 觀察 : 使用輪廓分析](homework/Day_056_kmean_HW.ipynb)
`非監督模型要以特殊評估方法(而非評估函數)來衡量, 今日介紹大家了解並使用其中一種方法 : 輪廓分`
57. [clustering 3 階層分群算法](homework/Day_057_HW.ipynb)
`hierarchical clustering`
58. [階層分群法 觀察 : 使用 2D 樣版資料集](homework/Day_058_hierarchical_clustering_HW.ipynb)
`非監督評估方法 : 2D樣版資料集是什麼? 如何生成與使用?`
59. [dimension reduction 1 降維方法-主成份分析](homework/Day_059_HW.ipynb)
`PCA`
60. [PCA 觀察 : 使用手寫辨識資料集](homework/Day_060_PCA_HW.ipynb)
`以較複雜的範例 : sklearn版手寫辨識資料集, 展示PCA的降維與資料解釋能力`
61. [dimension reduction 2 降維方法-T-SNE](homework/Day_061_HW.ipynb)
`TSNE`
62. [t-sne 觀察 : 分群與流形還原](homework/Day_062_tsne_HW.ipynb)
`什麼是流形還原? 除了 t-sne 之外還有那些常見的流形還原方法?`
## 7. 深度學習理論與實作
> 神經網路的運用
63. [神經網路介紹](homework/Day_063_HW.ipynb)
`Neural Network 簡介`
64. [深度學習體驗 : 模型調整與學習曲線](homework/Day_064_HW.ipynb)
`介紹體驗平台 TensorFlow PlayGround，並初步了解模型的調整`
65. [深度學習體驗 : 啟動函數與正規化](homework/Day_065_HW.ipynb)
`在 TF PlayGround 上，體驗進階版的深度學習參數調整`
## 8. 初探深度學習使用Keras
> 學習機器學習(ML)與深度學習( DL) 的好幫手
66. [Keras 安裝與介紹](homework/Day_066_Keras_Introduction_HW.ipynb)
`如何安裝 Keras 套件`
67. [Keras Dataset](homework/Day_067-Keras_Dataset_HW.ipynb)
`Keras embedded dataset的介紹與應用`
68. [Keras Sequential API](homework/Day_068_Keras_Sequential_Model_HW.ipynb)
`序列模型搭建網路`
69. [Keras Module API](homework/Day_069-keras_Module_API_HW.ipynb)
`Keras Module API的介紹與應用`
70. [Multi-layer Perception多層感知](homework/Day_070_Keras_Mnist_MLP_HW.ipynb)
`MLP簡介`
71. [損失函數](homework/Day_071_%E4%BD%BF%E7%94%A8%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8_HW.ipynb)
`損失函數的介紹與應用`
72. [啟動函數](homework/Day_072_Activation_function_HW.ipynb)
`啟動函數的介紹與應用`
73. [梯度下降Gradient Descent](homework/Day_073_Gradient_Descent_HW.ipynb)
`梯度下降Gradient Descent簡介`
74. [Gradient Descent 數學原理](homework/Day_074_Gradient_Descent_HW.ipynb)
`介紹梯度下降的基礎數學原理`
75. [BackPropagation](homework/Day_075_Back_Propagation_HW.ipynb)
`反向式傳播簡介`
76. [優化器optimizers](homework/Day_076_optimizer_HW.ipynb)
`優化器optimizers簡介`
77. [訓練神經網路的細節與技巧 - Validation and overfit](homework/Day_077_HW.ipynb)
`檢視並了解 overfit 現象`
78. [訓練神經網路前的注意事項](homework/Day_078_HW.ipynb)
`資料是否經過妥善的處理？運算資源為何？超參數的設置是否正確？`
79. [訓練神經網路的細節與技巧 - Learning rate effect](homework/Day_079_HW.ipynb)
`比較不同 Learning rate 對訓練過程及結果的差異`
80. [\[練習 Day\] 優化器與學習率的組合與比較]((homework/Day_080_HW.ipynb))
`練習時間：搭配不同的優化器與學習率進行神經網路訓練`
81. [訓練神經網路的細節與技巧 - Regularization](homework/Day_081_HW.ipynb)
`因應 overfit 的方法概述 - 正規化 (Regularization)`
82. [訓練神經網路的細節與技巧 - Dropout](homework/Day_082_HW.ipynb)
`因應 overfit 的方法概述 - 隨機缺失 (Dropout)`
83. [訓練神經網路的細節與技巧 - Batch normalization](homework/Day_083_HW.ipynb)
`因應 overfit 的方法概述 - 批次正規化 (Batch Normalization)`
84. [\[練習 Day\] 正規化/機移除/批次標準化的 組合與比較](homework/Day_084_HW.ipynb)
`練習時間：Hyper-parameters 大雜燴`
85. [訓練神經網路的細節與技巧 - 使用 callbacks 函數做 earlystop](homework/Day_085_HW.ipynb)
`因應 overfit 的方法概述 - 悔不當初的煞車機制 (EarlyStopping)`
86. [訓練神經網路的細節與技巧 - 使用 callbacks 函數儲存 model](homework/Day_086_HW.ipynb)
`使用 Keras 內建的 callback 函數儲存訓練完的模型`
87. [訓練神經網路的細節與技巧 - 使用 callbacks 函數做 reduce learning rate](homework/Day_087_HW.ipynb)
`使用 Keras 內建的 callback 函數做學習率遞減`
88. [訓練神經網路的細節與技巧 - 撰寫自己的 callbacks 函數](homework/Day_088_HW.ipynb)
89. [訓練神經網路的細節與技巧 - 撰寫自己的 Loss function](homework/Day_089_HW.ipynb)
`瞭解如何撰寫客製化的損失函數，並用在模型訓練上`
90. [使用傳統電腦視覺與機器學習進行影像辨識](homework/Day_090_color_histogram_HW.ipynb)
`了解在神經網路發展前，如何使用傳統機器學習演算法處理影像辨識`
91. [\[練習 Day\] 使用傳統電腦視覺與機器學習進行影像辨識](homework/Day_091_classification_with_cv_HW.ipynb)
`應用傳統電腦視覺方法＋機器學習進行 CIFAR-10 分類`
## 9. 深度學習應用卷積神經網路
> 卷積神經網路(CNN)常用於影像辨識的各種應用，譬如醫療影像與晶片瑕疵檢測
92. [卷積神經網路 (Convolution Neural Network, CNN) 簡介](homework/Day_092_CNN_theory.ipynb)
`了解CNN的重要性, 以及CNN的組成結構`
93. [卷積神經網路架構細節](homework/Day_093_CNN_Brief_HW.ipynb)
`為什麼比DNN更適合處理影像問題, 以及Keras上如何實作CNN`
94. [卷積神經網路 - 卷積(Convolution)層與參數調整](homework/Day_094_CNN_Convolution_HW.ipynb)
`卷積層原理與參數說明`
95. [卷積神經網路 - 池化(Pooling)層與參數調整](homework/Day_095_CNN_Pooling_Padding_HW.ipynb)
`池化層原理與參數說明`
96. [Keras 中的 CNN layers](homework/Day_096_Keras_CNN_layers.ipynb)
`介紹 Keras 中常用的 CNN layers`
97. [使用 CNN 完成 CIFAR-10 資料集](homework/Day_097_Keras_CNN_vs_DNN.ipynb)
`透過 CNN 訓練 CIFAR-10 並比較其與 DNN 的差異`
98. [訓練卷積神經網路的細節與技巧 - 處理大量數據](homework/Day_098_Python_generator.ipynb)
`資料無法放進記憶體該如何解決？如何使用 Python 的生成器 generator?`
99. [訓練卷積神經網路的細節與技巧 - 處理小量數據](homework/Day_099_data_augmentation.ipynb)
`資料太少準確率不高怎麼辦？如何使用資料增強提升準確率？`
100. [訓練卷積神經網路的細節與技巧 - 轉移學習 (Transfer learning)](homework/Day_100_transfer_learning_HW.ipynb)
`何謂轉移學習 Transfer learning？該如何使用？`
