from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and average price5
with open('model.pkl','rb') as f:
    model = pickle.load(f)

with open('average_price.pkl','rb') as f:
    average_price = pickle.load(f)

def convert_m2_to_sqft(m2):
    return m2 * 510.7639

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    transaction_type = request.form.get('transaction_type')
    try:
        # Nhận các dữ liệu từ form
        floors = float(request.form.get('floors'))
        waterfront = int(request.form.get('waterfront'))
        bedrooms = int(request.form.get('bedrooms'))
        view = int(request.form.get('view'))
        bathrooms = float(request.form.get('bathrooms'))  
        sqft_above_m2 = float(request.form.get('sqft_above'))
        sqft_above = convert_m2_to_sqft(sqft_above_m2)       
        grade = int(request.form.get('grade'))      
        sqft_living_m2 = float(request.form.get('sqft_living'))
        sqft_living = convert_m2_to_sqft(sqft_living_m2)

        # Chuẩn bị dữ liệu dự đoán
        features_dict = {
            'floors': [floors],
            'waterfront': [waterfront],
            'bedrooms': [bedrooms],
            'view': [view],
            'bathrooms': [bathrooms],
            'sqft_above': [sqft_above],
            'grade': [grade],
            'sqft_living': [sqft_living]
        }
        features = pd.DataFrame(features_dict)

        # Dự đoán giá
        predicted_price = model.predict(features)[0]

        # In kết quả giá dự đoán ra terminal
        print(f"Predicted Price: {predicted_price}")
        
        # Đảm bảo giá dự đoán không âm
        if predicted_price < 0:
            predicted_price = 0

        # Sử dụng giá trị trung bình để đưa ra gợi ý
        if transaction_type == "buy":
            recommendation = "buy" if predicted_price < average_price else "don't buy"
        elif transaction_type == "sell":
            recommendation = "sell" if predicted_price > average_price else "don't sell"
        else:
            recommendation = "no recommendation"

        print(f"Predicted price: {predicted_price}, Average price: {average_price}, Recommendation: {recommendation}")

        return render_template('result.html', price=predicted_price, recommendation=recommendation)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
