
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. 기본 설정
# ==============================

DATA_PATH = "data/smartphone_usage_sleep_dataset.csv"
RESULT_PATH = "results"

os.makedirs(RESULT_PATH, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==============================
# 2. 데이터 불러오기
# ==============================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n[데이터 미리보기]")
    print(df.head())

    print("\n[데이터 정보]")
    print(df.info())

    return df


# ==============================
# 3. 데이터 전처리
# ==============================

def preprocess_data(df):

    # 분석에 사용할 변수
    features = [
        "Smartphone_Usage_Hours"
    ]

    target = "Sleep_Duration_Hours"

    # 필요한 컬럼 확인
    required_columns = features + [target]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    data = df[required_columns].copy()

    # 숫자형 변환
    for col in required_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    # 결측치 제거
    data = data.dropna()

    # 비정상적인 값 제거
    data = data[
        (data["Smartphone_Usage_Hours"] >= 0) &
        (data["Smartphone_Usage_Hours"] <= 24) &
        (data["Sleep_Duration_Hours"] >= 0) &
        (data["Sleep_Duration_Hours"] <= 24)
    ]

    print("\n[전처리 완료]")
    print("데이터 개수:", len(data))

    return data


# ==============================
# 4. 상관관계 분석
# ==============================

def correlation_analysis(data):

    correlation = data.corr()

    print("\n[상관관계 분석]")
    print(correlation)

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Smartphone Usage and Sleep Correlation")
    plt.tight_layout()

    plt.savefig(
        f"{RESULT_PATH}/correlation.png"
    )

    plt.close()


# ==============================
# 5. 모델 학습
# ==============================

def train_models(data):

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

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test, predictions
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(
            y_test, predictions
        )

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"\n[{name} 평가 결과]")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2 Score: {r2:.3f}")

        # 회귀 그래프
        if name == "Linear Regression":

            plt.figure(figsize=(8, 5))

            plt.scatter(
                X_test.iloc[:, 0],
                y_test,
                label="Actual"
            )

            plt.scatter(
                X_test.iloc[:, 0],
                predictions,
                label="Predicted"
            )

            plt.xlabel("Smartphone Usage (Hours)")
            plt.ylabel("Sleep Duration (Hours)")
            plt.title("Actual vs Predicted Sleep Duration")
            plt.legend()

            plt.tight_layout()

            plt.savefig(
                f"{RESULT_PATH}/regression.png"
            )

            plt.close()

    return results


# ==============================
# 6. 결과 저장
# ==============================

def save_results(results):

    result_df = pd.DataFrame(results).T

    result_df.to_csv(
        f"{RESULT_PATH}/model_evaluation.csv"
    )

    print("\n[모델 평가 결과 저장 완료]")
    print(result_df)


# ==============================
# 7. 프로그램 실행
# ==============================

def main():

    print("=" * 50)
    print("Smartphone Usage and Sleep Analysis")
    print("=" * 50)

    df = load_data()

    data = preprocess_data(df)

    correlation_analysis(data)

    results = train_models(data)

    save_results(results)

    print("\n분석이 완료되었습니다.")
    print("결과 저장 위치:", RESULT_PATH)


if __name__ == "__main__":
    main()
