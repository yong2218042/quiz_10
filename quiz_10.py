import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

while True:
    # Load the dataset
    url = "C:/Users/user/Downloads/09_irisdata.csv"  # 실제 CSV 파일의 경로로 대체
    column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    df = pd.read_csv(url, names=column_names)

    # Display the shape of the dataset
    print("데이터의 형태:", df.shape)

    # Display the summary statistics of the dataset
    print("데이터의 요약 통계:")
    print(df.describe())

    # Display the class distribution
    print("\n클래스 분포:")
    print(df.groupby('class').size())

    # Create a scatter matrix plot
    scatter_matrix(df, alpha=0.8, figsize=(10, 10), diagonal='hist')
    plt.show()

    # Split the data into independent variables (X) and the target variable (Y)
    X = df.iloc[:, :-1].values  # 독립 변수 (sepal-length, sepal-width, petal-length, petal-width)
    Y = df.iloc[:, -1].values   # 종속 변수 (class)

    # Train a Decision Tree model
    model = DecisionTreeClassifier(random_state=42)

    # K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

    # Display the average accuracy across folds
    print("\nK-Fold 교차 검증 평균 정확도:", results.mean())

    # 추가 입력을 받아 계속할지 여부 확인
    continue_input = input("\n더 진행하시겠습니까? (y/n): ").lower()
    if continue_input != 'y':
        print("프로그램을 종료합니다.")
        break

