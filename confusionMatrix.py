import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i]
        acc = sum_TP/np.sum(self.matrix)
        print("the model accuracy is ",acc)

        table = PrettyTable()
        table.field_names = ["","Precision","Recall","Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i,i]
            FP = np.sum(self.matrix[i,:])-TP
            FN = np.sum(self.matrix[:,i])-TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP/(TP+FP),3) if TP+FP!=0 else 0.
            Recall = round(TP/(TP+FN),3) if TP+FN!=0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i],Precision,Recall,Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix,cmap=plt.cm.Blues)

        # 设置坐标轴
        plt.xticks(range(self.num_classes),self.labels,rotation=45)
        plt.yticks(range(self.num_classes),self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max()/2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y,x])
                plt.text(x,y,info,verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black"
                         )
        plt.tight_layout()
        plt.show()
