import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

bundle_path = pathlib.Path(__file__).parent / "model_bundle.pkl"

@st.cache_resource
def load_bundle():
    with open(bundle_path, "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
ohe_cols = bundle["ohe_columns"]
medians = bundle["medians"]
cat_cols = bundle["categorical_cols"]
df_train = bundle["df_train"]

def to_number(series):
    return pd.to_numeric(series.astype(str).str.extract(r"([0-9]+\.?[0-9]*)",expand=False),errors="coerce")

def split_torque(series):
    raw=series.astype(str)
    torque_val=to_number(raw)
    torque_val=np.where(raw.str.lower().str.contains("kgm",na=False),torque_val*9.80665,torque_val)
    rpm_val=raw.str.extract(r"@\s*([0-9]+\s*-\s*[0-9]+|[0-9]+)",expand=False)
    rpm_val=(
        rpm_val.str.replace(" ","",regex=False)
        .str.split("-", expand=False)
        .apply(lambda x:np.nan if not isinstance(x,list) else(float(x[0])+float(x[-1]))/2)
    )
    return pd.to_numeric(torque_val,errors="coerce").astype(float),pd.to_numeric(rpm_val,errors="coerce").astype(float)

def preprocess(df):
    df = df.copy()

    for col in("mileage","engine","max_power"):
        if col in df.columns:
            df[col] = to_number(df[col])
    if "torque" in df.columns:
        df["torque"],df["max_torque_rpm"] = split_torque(df["torque"])

    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.split().str[0]

    df_ohe = pd.get_dummies(df,columns=cat_cols+["seats"],drop_first=True)
    df_ohe = df_ohe.reindex(columns=ohe_cols,fill_value=0)

    return scaler.transform(df_ohe)

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Car Price Predictor")

tab_eda, tab_pred, tab_weights = st.tabs(["📊 EDA", "🔮 Предсказание", "⚖️ Веса модели"])

with tab_eda:
    st.header("EDA")
    st.markdown(f"**Размер датасета:** {df_train.shape[0]:,} строк × {df_train.shape[1]} столбцов")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение цены (selling_price)")
        fig, ax = plt.subplots()
        ax.hist(df_train["selling_price"],bins=60,color="#4C72B0",edgecolor="white")
        ax.set_xlabel("Цена")
        ax.set_ylabel("Количество")
        ax.set_title("selling_price")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Год выпуска (year)")
        fig, ax = plt.subplots()
        ax.hist(df_train["year"],bins=30,color="#DD8452",edgecolor="white")
        ax.set_xlabel("Год")
        ax.set_ylabel("Количество")
        ax.set_title("year")
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("km_driven vs selling_price")
        fig, ax = plt.subplots()
        ax.scatter(df_train["km_driven"],df_train["selling_price"],alpha=0.3,s=8,color="#55A868")
        ax.set_xlabel("Пробег (км)")
        ax.set_ylabel("Цена")
        ax.set_title("km_driven vs selling_price")
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("max_power vs selling_price")
        fig, ax = plt.subplots()
        ax.scatter(df_train["max_power"],df_train["selling_price"],alpha=0.3,s=8,color="#C44E52")
        ax.set_xlabel("Мощность (bhp)")
        ax.set_ylabel("Цена")
        ax.set_title("max_power vs selling_price")
        st.pyplot(fig)
        plt.close(fig)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Тип топлива (fuel)")
        fuel_counts=df_train["fuel"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(fuel_counts.index, fuel_counts.values, color="#8172B2")
        ax.set_xlabel("Тип топлива")
        ax.set_ylabel("Количество")
        st.pyplot(fig)
        plt.close(fig)

    with col6:
        st.subheader("Корреляция признаков (Pearson)")
        corr=df_train.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title("Pearson correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_pred:
    st.header("Предсказание цены автомобиля")

    input_mode = st.radio("Способ ввода:", ["Ручной ввод", "Загрузить CSV"])

    if input_mode == "Ручной ввод":
        st.subheader("Введите характеристики автомобиля")

        c1, c2, c3 = st.columns(3)
        with c1:
            name=st.text_input("Название (name)",value="Maruti Swift")
            year=st.number_input("Год (year)",min_value=1990,max_value=2024,value=2015)
            km_driven=st.number_input("Пробег (km_driven)",min_value=0,value=50000)
        with c2:
            fuel=st.selectbox("Топливо (fuel)",["Petrol","Diesel","CNG","LPG","Electric"])
            seller=st.selectbox("Продавец (seller_type)",["Individual","Dealer","Trustmark Dealer"])
            transmission=st.selectbox("КПП (transmission)",["Manual","Automatic"])
        with c3:
            owner=st.selectbox("Владелец (owner)",["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"])
            mileage=st.number_input("Расход (mileage, kmpl)",min_value=0.0,value=20.0)
            engine=st.number_input("Объём (engine, cc)",min_value=0,value=1200)
        c4, c5 = st.columns(2)
        with c4:
            max_power=st.number_input("Мощность (max_power, bhp)",min_value=0.0,value=80.0)
            torque=st.text_input("Момент (torque)",value="110Nm@ 4200rpm")
        with c5:
            seats=st.selectbox("Число мест (seats)",[2,4,5,6,7,8,9,10],index=2)

        if st.button("Предсказать цену"):
            row=pd.DataFrame([{
                "name": name, "year": year, "km_driven": km_driven,
                "fuel": fuel, "seller_type": seller, "transmission": transmission,
                "owner": owner, "mileage": mileage, "engine": engine,
                "max_power": max_power, "torque": torque, "seats": seats,
                "max_torque_rpm": np.nan,
            }])
            try:
                X=preprocess(row)
                pred=model.predict(X)[0]
                st.success(f"**Предсказанная цена: ₹{pred:,.0f} (INR)**")
            except Exception as e:
                st.error(f"Ошибка предсказания: {e}")

    else:
        st.subheader("Загрузите CSV-файл")
        st.markdown("Файл должен содержать те же столбцы, что и train, но без `selling_price`.")
        uploaded = st.file_uploader("Выберите CSV",type=["csv"])
        if uploaded:
            try:
                df_up=pd.read_csv(uploaded)
                st.write("Первые строки:", df_up.head())
                X=preprocess(df_up)
                preds=model.predict(X)
                df_up["predicted_price"] = preds.astype(int)
                st.success(f"Предсказания готовы для {len(df_up)} объектов")
                st.dataframe(df_up[["name", "year", "km_driven", "predicted_price"]].head(50))
                csv_out=df_up.to_csv(index=False).encode("utf-8")
                st.download_button("Скачать результат", csv_out, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Ошибка обработки файла: {e}")

with tab_weights:
    st.header("Веса модели")

    coef_series = pd.Series(model.coef_,index=ohe_cols).sort_values()

    n_show = st.slider("Сколько признаков показать (топ N по |весу|)",10,len(ohe_cols),20)
    top_coef = coef_series.reindex(coef_series.abs().nlargest(n_show).index).sort_values()

    fig, ax = plt.subplots(figsize=(9,max(4,n_show*0.35)))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in top_coef.values]
    ax.barh(top_coef.index, top_coef.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент")
    ax.set_title(f"Топ-{n_show} признаков по |весу|")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Все коэффициенты (таблица)")
    df_coef=(coef_series.reset_index().rename(columns={"index":"feature",0:"coefficient"}).assign(abs_coef=lambda d:d["coefficient"].abs()).sort_values("abs_coef",ascending=False).drop(columns="abs_coef").reset_index(drop=True))
    st.dataframe(df_coef,use_container_width=True)
