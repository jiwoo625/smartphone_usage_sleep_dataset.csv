
import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =================================
# 1. Streamlit 기본 설정
# =================================

st.set_page_config(
    page_title="Smartphone Usage & Sleep Analysis",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Smartphone Usage and Sleep Analysis")
st.write(
    "스마트폰 사용 시간과 수면 시간의 관계를 분석하는 프로젝트입니다."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =================================
# 2. CSV 데이터 자동 탐색
# =================================

def find_csv():

    possible_paths = [
        os.path.join(
            BASE_DIR,
            "data",
            "smartphone_usage_sleep_dataset.csv"
        ),
        os.path.join(
            BASE_DIR,
            "smartphone_usage_sleep_dataset.csv"
        )
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    csv_files = glob.glob(
        os.path.join(BASE_DIR, "**", "*.csv"),
        recursive=True
    )

    if csv_files:
        return csv_files[0]

    return None


# =================================
# 3. 데이터 불러오기
# =================================

def load_data():

    csv_path = find_csv()

    if csv_path is None:

        st.error("CSV 데이터 파일을 찾을 수 없습니다.")

        st.info(
            "GitHub 저장소에 CSV 파일을 업로드하세요."
        )

        st.code(
            "data/smartphone_usage_sleep_dataset.csv"
        )

        return None

    try:

        df = pd.read_csv(csv_path)

        st.success("데이터를 성공적으로 불러왔습니다.")

        st.caption(f"데이터 위치: {csv_path}")

        return df

    except Exception as e:

        st.error(f"데이터 로드 오류: {e}")

        return None


# =================================
# 4. 데이터 전처리
# =================================

def preprocess_data(df):

    required_columns = [
        "Smartphone_Usage_Hours",
        "Sleep_Duration_Hours"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        st.error(
            f"CSV 컬럼명이 올바르지 않습니다: {missing}"
        )

        st.write("현재 CSV 컬럼:", list(df.columns))

        return None

    data = df[required_columns].copy()

    for col in required_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    data = data[
        (data["Smartphone_Usage_Hours"].between(0, 24)) &
        (data["Sleep_Duration_Hours"].between(0, 24))
    ]

    if len(data) < 5:

        st.error("분석할 데이터가 부족합니다. 최소 5개 이상의 유효한 행이 필요합니다.")

        return None

    return data


# =================================
# 5. 데이터 분석
# =================================

def show_data_analysis(data):

    st.header("1. Data Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Data",
        f"{len(data)}"
    )

    col2.metric(
        "Average Smartphone Usage",
        f"{data['Smartphone_Usage_Hours'].mean():.2f} hours"
    )

    col3.metric(
        "Average Sleep Duration",
        f"{data['Sleep_Duration_Hours'].mean():.2f} hours"
    )

    st.subheader("Dataset Preview")

    st.dataframe(data, use_container_width=True)

    st.subheader("Descriptive Statistics")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )


# =================================
# 6. 상관관계 분석
# =================================

def correlation_analysis(data):

    st.header("2. Correlation Analysis")

    correlation = data.corr().iloc[0, 1]

    st.metric(
        "Correlation Coefficient",
        f"{correlation:.3f}"
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Smartphone Usage and Sleep Correlation")

    st.pyplot(fig)

    plt.close(fig)

    if correlation < 0:

        st.write(
            "스마트폰 사용 시간과 수면 시간 사이에 음의 상관관계가 관찰됩니다."
        )

    elif correlation > 0:

        st.write(
            "스마트폰 사용 시간과 수면 시간 사이에 양의 상관관계가 관찰됩니다."
        )

    else:

        st.write("상관계수가 0에 가깝습니다.")


# =================================
# 7. 머신러닝 모델 학습 및 평가
# =================================

def train_models(data):

    st.header("3. Machine Learning")

    X = data[["Smartphone_Usage_Hours"]]

    y = data["Sleep_Duration_Hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    evaluation_results = {}

    predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, pred)
        )

        r2 = r2_score(y_test, pred)

        evaluation_results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        }

        predictions[name] = pred

    st.subheader("Model Evaluation")

    result_df = pd.DataFrame(
        evaluation_results
    ).T

    st.dataframe(
        result_df.style.format("{:.3f}"),
        use_container_width=True
    )

    st.download_button(
        "Download Evaluation Results",
        result_df.to_csv().encode("utf-8-sig"),
        file_name="model_evaluation.csv",
        mime="text/csv"
    )

    return models, predictions, X_test, y_test, result_df


# =================================
# 8. 예측 결과 시각화
# =================================

def show_prediction(models, predictions, X_test, y_test):

    st.header("4. Prediction Visualization")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        X_test["Smartphone_Usage_Hours"],
        y_test,
        label="Actual",
        alpha=0.7
    )

    ax.scatter(
        X_test["Smartphone_Usage_Hours"],
        predictions["Linear Regression"],
        label="Linear Regression",
        alpha=0.7
    )

    ax.set_xlabel("Smartphone Usage (Hours)")

    ax.set_ylabel("Sleep Duration (Hours)")

    ax.set_title("Actual vs Predicted Sleep Duration")

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)


# =================================
# 9. 스마트폰 사용 시간 기반 예측
# =================================

def predict_sleep(models):

    st.header("5. Sleep Duration Prediction")

    usage = st.slider(
        "Select Smartphone Usage Time (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    input_data = pd.DataFrame({
        "Smartphone_Usage_Hours": [usage]
    })

    col1, col2 = st.columns(2)

    with col1:

        linear_result = models[
            "Linear Regression"
        ].predict(input_data)[0]

        st.metric(
            "Linear Regression Prediction",
            f"{linear_result:.2f} hours"
        )

    with col2:

        rf_result = models[
            "Random Forest"
        ].predict(input_data)[0]

        st.metric(
            "Random Forest Prediction",
            f"{rf_result:.2f} hours"
        )

    st.caption(
        "예측값은 학습 데이터에 기반한 추정치이며 실제 수면 시간을 보장하지 않습니다."
    )


# =================================
# 10. 메인 실행
# =================================

def main():

    df = load_data()

    if df is None:
        st.stop()

    data = preprocess_data(df)

    if data is None:
        st.stop()

    show_data_analysis(data)

    correlation_analysis(data)

    models, predictions, X_test, y_test, results = train_models(data)

    show_prediction(
        models,
        predictions,
        X_test,
        y_test
    )

    st.success("Analysis Completed!")

    st.caption(
        "This project analyzes the relationship between smartphone usage and sleep duration."
    )


if __name__ == "__main__":
    main()
