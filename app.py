import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Dự đoán giá xe cũ - Nhóm 2", page_icon="🚗")

@st.cache_resource
def tai_mo_hinh():
    # Huấn luyện Random Forest từ tập train (chỉ chạy 1 lần khi app khởi động)
    train = pd.read_excel("Nhom2_GiaXeCu_train.xlsx")
    X, y = train.drop(columns="price"), train["price"]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                  max_features=0.5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, list(X.columns)

model, cot = tai_mo_hinh()

HANG = ["BMW", "Chevrolet", "Ford", "Jeep", "Lexus", "Mercedes-Benz",
        "Porsche", "Toyota", "Other", "Hãng còn lại"]
NHIEN_LIEU = ["Gasoline", "E85 Flex Fuel", "Hybrid", "Plug-In Hybrid", "Unknown", "Khác"]
HOP_SO = ["Tự động", "Manual", "Other"]

st.title("🚗 DỰ ĐOÁN GIÁ XE CŨ - NHÓM 2")
st.write("Mô hình Random Forest. Nhập thông tin xe để nhận giá tham khảo (USD).")

c1, c2 = st.columns(2)
with c1:
    nam = st.number_input("Năm sản xuất", min_value=1974, max_value=2024, value=2019, step=1)
    so_dam = st.number_input("Số dặm đã đi (milage)", min_value=0, max_value=400000, value=40000, step=1000)
    ma_luc = st.number_input("Mã lực (hp)", min_value=70, max_value=1020, value=300, step=10)
    dung_tich = st.number_input("Dung tích động cơ (lít)", min_value=0.6, max_value=8.5, value=3.5, step=0.1)
with c2:
    hang = st.selectbox("Hãng xe", HANG, index=HANG.index("Toyota"))
    nhien_lieu = st.selectbox("Loại nhiên liệu", NHIEN_LIEU)
    hop_so = st.selectbox("Hộp số", HOP_SO)
    tai_nan = st.radio("Từng tai nạn", ["Không", "Có"], horizontal=True)
    giay_to = st.radio("Giấy tờ (clean title)", ["Sạch", "Không sạch"], horizontal=True)

if st.button("Dự đoán giá"):
    xe = dict.fromkeys(cot, 0)
    xe.update({"model_year": nam, "milage": so_dam, "hp": ma_luc, "liter": dung_tich,
               "accident": 1 if tai_nan == "Có" else 0,
               "clean_title": 1 if giay_to == "Sạch" else 0})
    for tien_to, gia_tri in [("brand_", hang), ("fuel_type_", nhien_lieu), ("transmission_", hop_so)]:
        if tien_to + gia_tri in xe:
            xe[tien_to + gia_tri] = 1
    gia = model.predict(pd.DataFrame([xe])[cot])[0]
    st.success(f"Giá dự đoán: {gia:,.0f} USD")
    st.caption("Đây là giá tham khảo, sai số trung bình khoảng 7.600 USD (MAE trên tập test).")
