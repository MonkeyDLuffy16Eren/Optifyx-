from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your model (Make sure 'sales_model.pkl' is the correct path to your saved model)
with open('sales_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Match the form field names from the HTML
            tv = request.form.get('tv')  # Get the 'tv' value from the form
            radio = request.form.get('radio')  # Get the 'radio' value from the form
            newspaper = request.form.get('newspaper')  # Get the 'newspaper' value from the form

            # Check if all inputs are provided
            if not tv or not radio or not newspaper:
                return render_template('index.html', prediction_text="Please fill out all fields.")

            # Convert input values to float
            tv = float(tv)
            radio = float(radio)
            newspaper = float(newspaper)

            # Debugging: Print form values to check if inputs are captured correctly
            print(f"TV: {tv}, Radio: {radio}, Newspaper: {newspaper}")

            # Prepare the input features for prediction
            features = np.array([[tv, radio, newspaper]])

            # Predict the sales
            prediction = model.predict(features)

            # Debugging: Print prediction to verify
            print(f"Prediction: {prediction[0]}")

            # Return the prediction result to the HTML template
            return render_template('index.html', prediction_text=f"Predicted Sales: {prediction[0]:.2f} Millions")

        except Exception as e:
            # If an error occurs, print it for debugging
            print(f"Error occurred: {e}")
            return render_template('index.html', prediction_text="Error in making prediction. Please check your input values.")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)