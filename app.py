from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load data
try:
    data = pd.read_csv("updated_customer_data_with_clusters.csv")
    with open('data.json', 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
except Exception as e:
    print(f"Error loading data: {e}")
    data = pd.DataFrame()
    cluster_data = {"clusters": []}


features = ["Income", "ShoppingFrequency", "AnnualExpenditure", "Age"]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

def format_currency(value):
    try:
        return "{:,.0f}".format(float(value))
    except:
        return "0"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        try:
            
            customer_data = {
                "Income": float(request.form["income"]),
                "ShoppingFrequency": int(request.form["shoppingFrequency"]),
                "AnnualExpenditure": float(request.form["annual"]),
                "Age": int(request.form["age"]),
                "customer_name": request.form["customer"],
                "occupation": request.form.get("occupation", request.form.get("other", "")),
                "favorite_product": request.form["favoriteProduct"]
            }

            # Validate data
            if any(not customer_data[field] for field in features):
                return render_template('index.html', error="Vui lòng nhập đầy đủ thông tin!")

            
            input_data = np.array([[
                customer_data[feature] for feature in features
            ]])
            input_scaled = scaler.transform(input_data)
            
            # Find nearest cluster
            distances = np.linalg.norm(data_scaled - input_scaled, axis=1)
            nearest_idx = distances.argmin()
            cluster = int(data.iloc[nearest_idx]["Cluster"])

            return redirect(url_for('result', 
                cluster=cluster,
                customer_name=customer_data["customer_name"],
                age=customer_data["Age"],
                occupation=customer_data["occupation"],
                income=customer_data["Income"],
                annual=customer_data["AnnualExpenditure"],
                shopping_frequency=customer_data["ShoppingFrequency"],
                favorite_product=customer_data["favorite_product"]
            ))

        except Exception as e:
            return render_template('index.html', error=f"Lỗi xử lý: {str(e)}")

    return render_template('index.html')

@app.route('/result')
def result():
    try:
        # Get all parameters with default values
        cluster_num = int(request.args.get('cluster', 0))
        customer_name = request.args.get('customer_name', '')
        age = request.args.get('age', 0)
        occupation = request.args.get('occupation', '')
        income = request.args.get('income', 0)
        annual = request.args.get('annual', 0)
        shopping_frequency = request.args.get('shopping_frequency', 0)
        favorite_product = request.args.get('favorite_product', '')

        # Get cluster info
        cluster_info = next(
            (c for c in cluster_data['clusters'] if c['Cluster'] == cluster_num),
            None
        )

        # Get occupation strategy
        occupation_strategy = None
        if cluster_info and occupation:
            occupation_strategy = cluster_info['Occupation Strategies'].get(
                occupation,
                "Chưa có chiến lược cụ thể cho nghề nghiệp này"
            )

        return render_template(
            'result.html',
            customer_name=customer_name,
            age=age,
            occupation=occupation,
            income=income,
            annual=annual,
            shopping_frequency=shopping_frequency,
            favorite_product=favorite_product,
            cluster_info=cluster_info,
            occupation_strategy=occupation_strategy,
            format_currency=format_currency
        )

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug log
        return render_template('index.html', error=f"Lỗi hiển thị kết quả: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)