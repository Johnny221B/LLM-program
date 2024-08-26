import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(test_actuals1, test_predictions1, test_actuals2, test_predictions2, title, filename):
    plt.figure(figsize=(8, 6))
    
    # 绘制测试数据集的预测值和实际值
    plt.scatter(test_actuals1, test_predictions1, label='Testing Data Predictions 1', color='red')
    plt.scatter(test_actuals2, test_predictions2, label='Testing Data Predictions 2', color='purple')
    
    # 绘制完美预测的对角线
    all_actuals = np.concatenate([test_actuals1, test_actuals2])
    plt.plot([min(all_actuals), max(all_actuals)], 
             [min(all_actuals), max(all_actuals)], 
             color='orange', label='Perfect Prediction')
    
    # 在具有相同整数部分的实际时间的点之间绘制虚线
    unique_actuals = np.unique(np.floor(all_actuals))
    for actual in unique_actuals:
        test_pred1 = test_predictions1[np.floor(test_actuals1) == actual]
        test_pred2 = test_predictions2[np.floor(test_actuals2) == actual]
        
        if len(test_pred1) > 0 and len(test_pred2) > 0:
            plt.plot([actual, actual], [test_pred1[0], test_pred2[0]], 'k--', linewidth=0.5)
    
    plt.xlabel('Actual Running Time')
    plt.ylabel('Predicted Running Time')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


train_predictions = np.array([8198.384,1322.0336,28458.865,31896.18,11046.095,25411.396,-1441.6696,32104.793,11135.174,11762.289,1909.724,28690.658,20568.455,-1018.9352,32334.95,31740.72])

train_actuals = np.array([10061.487,1653.8646,23053.283,22309.98,8836.583,19234.0,1839.1302,37774.016,7541.934,12565.417,1200.722,31033.168,21068.562,548.5238,28862.42,51294.598])

train_predictions2 = np.array([15522.029,1927.0739,34972.867,35240.86,2357.2417,42443.43,38405.312,15858.063,2015.3727,9546.127,13354.667,38954.184,33082.14,62576.656,33962.81,35366.266,3345.6382,3045.1597,28622.453,31825.793,45044.863,46257.355,10716.893,11544.131,2559.9526,12176.247,29045.855,14658.316,35592.477,38476.383,-2779.3872,2332.142])

train_actuals2 = np.array([19157.0,2103.9995,42152.0,37774.016,1653.8649,64717.766,22309.98,17874.0,1839.1305,8836.584,11458.001,51294.6,21068.562,70738.92,33760.0,30051.0,2423.9995,1200.7203,23053.283,19234.0,48919.793,42498.492,12565.418,7541.934,2233.0015,10061.487,31033.168,15092.0,45572.0,28862.42,548.525,2760.9995])

plot_predictions(train_actuals,train_predictions,train_actuals2,train_predictions2,"Contradiction between two sets","train_contraduct_time.png")