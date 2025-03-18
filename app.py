import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Başlık
st.title("DİYABET")

# Veri yükleme
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])
if uploaded_file is not None:
    # Veri yükleme ve okuma
    df = pd.read_csv(uploaded_file, sep="\s+", header=None)
    columnHeads = ["Pregnancies", "Glucose Concentration", "Dia Blood Pressure", "Skinfold Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age", "Result"]
    df.columns = columnHeads

    st.write("Dataset:")
    st.write(df.head())

    # Histogramlar
    st.subheader("Histograms")
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i, column in enumerate(df.columns):
        axs[i//3, i%3].hist(df[column], bins=10, color='skyblue', edgecolor='black')
        axs[i//3, i%3].set_title(column)
        axs[i//3, i%3].set_xlabel(column)
        axs[i//3, i%3].set_ylabel('Frequency')
    st.pyplot(fig)

    # Min-Max normalizasyonu
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

    st.write("Min-Max Normalized Data:")
    st.write(normalized_df.head())

    # Özellikler ve hedef değişken
    X = normalized_df.drop('Result', axis=1)
    y = normalized_df['Result']

    # RFE ile özellik seçimi ve Logistic Regression modeli
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=3)
    X_rfe = rfe.fit_transform(X, y)

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)

    # Modeli eğit
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Confusion Matrix görselleştirmesi
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Model başarımını değerlendir
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")

    # Farklı karmaşıklık seviyeleri için modelleme ve overfitting değerlendirmesi
    X = df.drop('Result', axis=1)
    y = df['Result']
    X_normalized = scaler.fit_transform(X)
    fs_method = RFE(LogisticRegression(max_iter=1000), n_features_to_select=3)
    model = LogisticRegression(max_iter=1000)
    selected_X = fs_method.fit_transform(X_normalized, y)
    X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.2, random_state=42)
    
    train_scores = []
    test_scores = []
    complexity_levels = []

    for complexity in range(1, selected_X.shape[1] + 1):
        model.fit(X_train[:, :complexity], y_train)
        train_scores.append(model.score(X_train[:, :complexity], y_train))
        test_scores.append(model.score(X_test[:, :complexity], y_test))
        complexity_levels.append(complexity)

    fig, ax = plt.subplots()
    ax.plot(complexity_levels, train_scores, label='Training Accuracy', marker='o')
    ax.plot(complexity_levels, test_scores, label='Testing Accuracy', marker='o')
    ax.set_xlabel('Complexity Level (Number of Features)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overfitting Evaluation - RFE - Logistic Regression')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
