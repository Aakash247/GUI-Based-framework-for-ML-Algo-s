import datasets as datasets
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Visualization Using ML ALGOL's")
st.write("Which one is the most suitable")
dataset_name = st.sidebar.selectbox("Select Dataset",
                                    ("CO2_Emission_Canada", "Diabetes", "Drug200", "Iris1", "Cars",
                                     "Pre-Defined Dataset"))

if dataset_name == "Cars":
    classifier_name = st.sidebar.selectbox("Select Classifier", ("Hierarchical Clustering ",))
    import numpy as np
    import pandas as pd
    from scipy.cluster import hierarchy
    from scipy.spatial import distance_matrix
    from matplotlib import pyplot as plt
    from sklearn.cluster import AgglomerativeClustering

    filename = 'cars_clus.csv'
    pdf = pd.read_csv(filename)
    st.write("Shape Of Dataset:", pdf.shape)
    pdf[['sales', 'resale', 'type', 'price', 'engine_s',
         'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
         'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                                   'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                                   'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
    pdf = pdf.dropna()
    pdf = pdf.reset_index(drop=True)
    featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
    from sklearn.preprocessing import MinMaxScaler

    x = featureset.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    feature_mtx = min_max_scaler.fit_transform(x)
    import scipy

    leng = feature_mtx.shape[0]
    D = scipy.zeros([leng, leng])
    for i in range(leng):
        for j in range(leng):
            D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
    import pylab
    import scipy.cluster.hierarchy

    Z = hierarchy.linkage(D, 'complete')
    from scipy.cluster.hierarchy import fcluster

    max_d = 3
    clusters = fcluster(Z, max_d, criterion='distance')
    from scipy.cluster.hierarchy import fcluster

    k = 5
    clusters = fcluster(Z, k, criterion='maxclust')
    fig = pylab.figure(figsize=(18, 50))


    def llf(id):
        return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))


    dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
    st.pyplot()
    agglom = AgglomerativeClustering(n_clusters=5, linkage='complete')
    agglom.fit(D)

if dataset_name == "Drug200":
    classifier_name = st.sidebar.selectbox("Select Classifier", ("Decision Tree ",))
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    my_data = pd.read_csv("drug200.csv")
    size = st.sidebar.slider("Train set:", 0.01, 1.0)
    maxDepth = st.sidebar.slider("Max Depth", 1, 20)
    st.markdown("*")
    st.write("Algorithm Being Used: ", classifier_name)
    st.write("Shape Of Dataset:", my_data.shape)
    X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    from sklearn import preprocessing

    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])
    X[:, 1] = le_sex.transform(X[:, 1])
    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    X[:, 2] = le_BP.transform(X[:, 2])
    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    X[:, 3] = le_Chol.transform(X[:, 3])
    y = my_data["Drug"]
    from sklearn.model_selection import train_test_split

    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=1 - size, random_state=3)
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=maxDepth)
    drugTree.fit(X_trainset, y_trainset)
    predTree = drugTree.predict(X_testset)
    from sklearn import metrics
    import matplotlib.pyplot as plt

    st.write("Accuracy", (metrics.accuracy_score(y_testset, predTree)) * 100)
    from io import StringIO
    import pydotplus
    import matplotlib.image as mpimg
    from sklearn import tree

    dot_data = StringIO()
    filename = "drugtree.png"
    featureNames = my_data.columns[0:5]
    out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data,
                               class_names=np.unique(y_trainset), filled=True, special_characters=True, rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    st.image(img)
    plt.imshow(img, interpolation='nearest')
    st.pyplot()

if dataset_name == "CO2_Emission_Canada":
    classifier_name = st.sidebar.selectbox("Select Classifier", ("Simple Linear", "Multi-Linear"))
    import pandas as pd

    df = pd.read_csv("CO2 Emissions_Canada.csv")

if dataset_name == "Iris1":
    classifier_name = st.sidebar.selectbox("Select Classifier", ("K-Clustering",))
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    pd.plotting.register_matplotlib_converters()
    import sklearn.datasets as datasets
    from sklearn.cluster import KMeans

    irisData = datasets.load_iris()
    iris_df = pd.DataFrame(irisData.data, columns=irisData.feature_names)
    x = iris_df.iloc[:, [0, 1, 2, 3]].values
    kmeans = KMeans(n_clusters=3, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(x)
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
                s=100, c='red', alpha=0.4, label='Iris-setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
                s=100, c='orange', label='Iris-versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
                s=100, c='blue', alpha=0.6, label='Iris-virginica')
    plt.legend()
    st.pyplot()
    st.markdown("*")
    st.write("""# Elbow Method""")
    from sklearn.cluster import KMeans

    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    st.pyplot()

if dataset_name == "Diabetes":
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KMN", "SVM_"))
    import pandas as pd

    df = pd.read_csv("diabetes.csv")

if dataset_name == "Pre-Defined Dataset":
    dataset_name = st.sidebar.selectbox(
        'Select Dataset',
        ('Iris', 'Breast Cancer', 'Wine')
    )

    st.write(f"## {dataset_name} Dataset")

    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('KNN', 'SVM', 'Random Forest')
    )


    def get_dataset(name):
        data = None
        if name == 'Iris':
            data = datasets.load_iris()
        elif name == 'Wine':
            data = datasets.load_wine()
        else:
            data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
        return X, y


    X, y = get_dataset(dataset_name)
    st.write('Shape of dataset:', X.shape)
    st.write('number of classes:', len(np.unique(y)))


    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
        return params


    params = add_parameter_ui(classifier_name)


    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        else:
            clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                               max_depth=params['max_depth'], random_state=1234)
        return clf


    clf = get_classifier(classifier_name, params)
    #### CLASSIFICATION ####

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
                c=y, alpha=0.8,
                cmap='viridis')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    # plt.show()
    st.pyplot(fig)

if classifier_name == "Simple Linear":
    size = st.sidebar.slider("Train set:", 0.01, 1.0)
    st.markdown("*")
    st.write("Algorithm Being Used: ", classifier_name)
    st.write("Shape Of Dataset:", df.shape)
    import matplotlib.pyplot as plt

    X = df.iloc[:, 7].values
    Y = df.iloc[:, -1].values
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    from sklearn.preprocessing import StandardScaler

    scale = StandardScaler()
    X = scale.fit_transform(X.astype(float))
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - size, random_state=0)
    print('Train set:', X_train.shape, Y_train.shape)
    print('Test set:', X_test.shape, Y_test.shape)
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    from sklearn.metrics import r2_score

    st.write("Accuracy:", r2_score(Y_test, Y_pred) * 100)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    import seaborn as sns

    sns.heatmap(df.corr(), annot=True)
    st.pyplot()
    plt.scatter(X_train, Y_train, color='red')
    plt.plot(X_train, lr.predict(X_train), color='blue')
    plt.title('CO2 Emission VS Fuel Consumption City (L/100 km)')
    plt.xlabel('Fuel Consumption City (L/100 km)')
    plt.ylabel('CO2 Emission')
    plt.show()
    st.pyplot()

if classifier_name == "Multi-Linear":
    size = st.sidebar.slider("Train set:", 0.01, 1.0)
    st.markdown("*")
    st.write("Algorithm Being Used: ", classifier_name)
    st.write("Shape Of Dataset:", df.shape)
    df.columns = ['Make', 'Model', 'VehicleClass', 'EngineSize(L)', 'Cylinders', 'Transmission', 'FuelType',
                  'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
                  'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)']
    import seaborn as sns

    sns.heatmap(df.corr(), annot=True)
    X = df[
        ['EngineSize(L)', 'Transmission', 'FuelType', 'VehicleClass', 'Cylinders', 'Fuel Consumption City (L/100 km)',
         'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']].values
    Y = df.iloc[:, -1].values
    Y = Y.reshape(-1, 1)
    from sklearn.preprocessing import LabelEncoder

    labelencoder_X = LabelEncoder()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    from sklearn.preprocessing import StandardScaler

    scale = StandardScaler()
    X = scale.fit_transform(X.astype(float))
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - size, random_state=0)
    print('Train set:', X_train.shape, Y_train.shape)
    print('Test set:', X_test.shape, Y_test.shape)
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    from sklearn.metrics import r2_score

    st.write("Accuracy:", r2_score(Y_test, Y_pred) * 100)

if classifier_name == "KMN":
    k = st.sidebar.slider("Number Of Neighbours:", 1, 20)
    st.markdown("*")
    st.write("Algorithm Being Used: ", classifier_name)
    st.write("Shape Of Dataset:", df.shape)
    col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for i in col:
        df[i].replace(0, df[i].mean(), inplace=True)
    X = df.iloc[:, 0:-1].values
    Y = df.iloc[:, -1].values
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit(X).transform(X.astype(float))
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print('Train set:', X_train.shape, Y_train.shape)
    print('Test set:', X_test.shape, Y_test.shape)
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    Y_pred = neigh.predict(X_test)
    from sklearn import metrics

    st.write("Accuracy:", metrics.accuracy_score(Y_test, Y_pred) * 100)
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import datasets, neighbors
    from mlxtend.plotting import plot_decision_regions
    from sklearn.decomposition import PCA
    from mlxtend.plotting import plot_decision_regions
    from sklearn.svm import SVC

    clf = SVC(C=100, gamma=0.0001)
    pca = PCA(n_components=2)
    X_train2 = pca.fit_transform(X)
    clf.fit(X_train2, df['Outcome'].astype(int).values)
    plot_decision_regions(X_train2, df['Outcome'].astype(int).values, clf=clf, legend=2)
    st.pyplot()
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    st.markdown("*")
    st.write("""    # The Confusion Matrix:    """)
    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True)
    st.pyplot()
    cm1 = confusion_matrix(Y_test, Y_pred)
    tp = cm1[1][1]
    tn = cm1[0][0]
    fp = cm1[0][1]
    fn = cm1[1][0]
    P = tp + fn
    N = tn + fp
    pr = tp / (fp + tp)
    rec = tp / (fn + tp)
    f1 = 2 * (pr * rec) / (pr + rec)
    sens = (tp / P)
    spec = (tn / N)
    fpr = fp / N
    fnr = fn / P
    npv = tn / (tn + fn)
    fdr = fp / (fp + tp)
    st.markdown("*")
    st.write("""    # ROC Curve:    """)
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('AUC: %.2f' % auc)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    st.pyplot()

if classifier_name == "SVM_":
    size = st.sidebar.slider("Train set:", 0.01, 1.0)
    st.markdown("*")
    st.write("Algorithm Being Used: ", classifier_name)
    st.write("Shape Of Dataset:", df.shape)
    col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for i in col:
        df[i].replace(0, df[i].mean(), inplace=True)
    X = df.iloc[:, 0:-1].values
    Y = df.iloc[:, -1].values
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit(X).transform(X.astype(float))
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - size, random_state=0)
    print('Train set:', X_train.shape, Y_train.shape)
    print('Test set:', X_test.shape, Y_test.shape)
    from sklearn.svm import SVC

    model = SVC(kernel='rbf', random_state=0).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score

    st.write("Accuracy:", accuracy_score(Y_test, Y_pred) * 100)
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2,
                c=Y, alpha=0.8,
                cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
    st.pyplot()
    st.markdown("*")
    st.write("""    # The Confusion Matrix:    """)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True)
    st.pyplot()
    cm1 = confusion_matrix(Y_test, Y_pred)
    tp = cm1[1][1]
    tn = cm1[0][0]
    fp = cm1[0][1]
    fn = cm1[1][0]
    P = tp + fn
    N = tn + fp
    pr = tp / (fp + tp)
    rec = tp / (fn + tp)
    f1 = 2 * (pr * rec) / (pr + rec)
    sens = (tp / P)
    spec = (tn / N)
    fpr = fp / N
    fnr = fn / P
    npv = tn / (tn + fn)
    fdr = fp / (fp + tp)
    st.markdown("*")
    st.write("""    # ROC Curve:    """)
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('AUC: %.2f' % auc)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    st.pyplot()