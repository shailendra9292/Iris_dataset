
import sys
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import  DecisionTreeClassifier
import matplotlib.patches as mpatches


def models(n,X_train,y_train,X_test,y_test):
    choice=int(input('enter 1 to visualize training set and 2 for test set\t'))
    if choice ==1: 
        X=X_train
        y=y_train
    else :
        X=X_test
        y=y_test
        
    # Logistic Regression
    if n==1:
        clf = LogisticRegression(C=100, random_state = 0, solver='newton-cg',multi_class='multinomial')
        clf.fit(X,y)
        visualization(X,y,n,clf)
       
    if n==2:
        # KNN
        clf = KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')
        clf.fit(X,y)
        visualization(X,y,n,clf)

    if n==3:
        # SVM
        clf = SVC(C=10,gamma = 0.9,kernel='rbf',random_state=0)
        clf.fit(X,y)
        visualization(X,y,n,clf)

    if n==4:
        # Decision tree 
        clf =  DecisionTreeClassifier(criterion='gini',random_state=0)
        clf.fit(X,y)
        visualization(X,y,n,clf)

    if n==5:
        # Naive bayes
        clf = GaussianNB()
        clf.fit(X,y)
        visualization(X,y,n,clf)

    if n==6:
        # Random Forest
        clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0,n_jobs= -1)
        clf.fit(X,y)
        visualization(X,y,n,clf)
    
    if n==7: sys.exit(0)
    
    y_pred=clf.predict(X)
    print(confusion_matrix(y,y_pred))

def visualization(X,y,i,clf):
    
    h=0.02
    x_min , x_max = X.min() -3 , X.max() +3
    y_min , y_max = y.min() -3, y.max() +3
    xx , yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.85)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=120,cmap=plt.cm.Spectral,edgecolor='k',
                alpha=0.9)
    plt.xlabel(' petal length(cm)')
    plt.ylabel('petal width (cm)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.colorbar(cmap=plt.cm.Spectral)
    plt.axis()
    plt.title(titles[i-1],fontsize=15,fontweight='bold')
    mediumvioletred = mpatches.Patch(color='mediumvioletred', label='Setosa')
    royalblue = mpatches.Patch(color='royalblue', label='Versicolor')
    blanchedalmond = mpatches.Patch(color='blanchedalmond', label='Virginica')
    plt.legend(handles=[mediumvioletred, royalblue, blanchedalmond])
    # accuracy

    score = clf.score(X,y)*100
    plt.text(xx.min() + .3, yy.max() - .3, ('%.2f %%' % score),fontsize=17,
             fontweight = 'bold',style='italic')
    plt.show()

if __name__=="__main__":
   

    # Data preprocessing
    dataset = load_iris()
    X_sepal = dataset.data[:,:2]
    X_petal = dataset.data[:,2:]
    y = dataset.target
    
    # feature scaling
    sc_s=StandardScaler()
    sc_p=StandardScaler()
    X_sepal=sc_s.fit_transform(X_sepal)
    X_petal=sc_p.fit_transform(X_petal)

    # Split dataset
    X_train_s , X_test_s , y_train_s , y_test_s = train_test_split(X_sepal, y,test_size = 0.25 ,
                   random_state = 0)
    X_train_p , X_test_p , y_train_p , y_test_p = train_test_split(X_petal, y, test_size = 0.25 ,
                   random_state = 0)
    
    titles = ['Logistic Regression','KNearNeighbour','Support Vector (RBF)','Decision Tree ',
              'Naive Bayes','Random Forest']
  
    n=int(input('enter 1 for sepal 2 for petal\n')) 
    print('enter 1 for logistic regression')
    print('enter 2 for knn')
    print('enter 3 for SVM')
    print('enter 4 for decision tree')
    print('enter 5 for naive bayes')
    print('enter 6 for random forest')
    print('enter 7 for exit')
    m=int(input('enter your choice\n'))
    # pass train/test variable
    if n==1:
        models(m,X_train_s,y_train_s,X_test_s,y_test_s)        
    else : 
        models(m,X_train_p,y_train_p,X_test_p,y_test_p)
