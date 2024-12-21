from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from sklearn.impute import SimpleImputer
import joblib
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Constants
m = 6.0  # power-law coefficient
kappa = 0.41  # von Karman constant
FS = 22  # Font size for plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': FS,
    'axes.labelsize': FS,
    'axes.titlesize': FS,
    'legend.fontsize': FS,
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'figure.dpi': 300
})


# Utility function: Calculate sediment properties
def calculate_sediment_properties(velocity, water_depth, diameter):
    shear_velocity = ((m + 1) / m) * velocity * kappa / m
    settling_velocity = diameter * 0.01  # Approximation for simplicity
    Ro_beta = min(3, 1 + 2 * (settling_velocity / shear_velocity) ** 2)
    Ro = settling_velocity / (Ro_beta * kappa * shear_velocity)
    return shear_velocity, settling_velocity, Ro


# Utility function: Dynamic model loader
def load_models(mode):
    if mode == "single":
        scaler = joblib.load('Particles_single_checkpoint/Particles_7features_scaler.pkl')
        rf_model = joblib.load('Particles_single_checkpoint/Particles_7features_rf_model.pkl')
        kan_model = load_model('Particles_single_checkpoint/Particles_7features_kan_model.keras')
    else:  # multiple
        scaler = joblib.load('Particles_single_checkpoint_1600/Particles_5features_scaler.pkl')
        rf_model = joblib.load('Particles_single_checkpoint_1600/Particles_5features_rf_model.pkl')
        kan_model = load_model('Particles_single_checkpoint_1600/Particles_5features_kan_model.keras')
    return scaler, rf_model, kan_model


@app.route("/", methods=["GET", "POST"])
def index():
    calculated_values, plot_filename = None, None
    velocity, water_depth, diameter = None, None, None
    mode = "single"

    if request.method == "POST":
        # Parse form inputs
        mode = request.form['mode']
        velocity = request.form.get('feature1')
        water_depth = request.form.get('feature2')
        diameter = request.form.get('feature3', "")

        try:
            velocity = float(velocity) if velocity else None
            water_depth = float(water_depth) if water_depth else None
            diameter = float(diameter) if diameter else None
        except ValueError:
            return "Please enter valid numeric values."

        if mode == "single" and velocity and water_depth and diameter:
            # Calculate sediment properties
            shear_velocity, settling_velocity, Ro = calculate_sediment_properties(velocity, water_depth, diameter)
            calculated_values = {'shear_velocity': shear_velocity, 'settling_velocity': settling_velocity, 'Ro': Ro}

            # Prepare features for prediction
            predict_features = np.array([velocity, water_depth, diameter, shear_velocity, settling_velocity, Ro]).reshape(1, -1)

        elif mode == "multiple" and velocity and water_depth:
            predict_features = np.array([[velocity, water_depth]])

        else:
            return "Please provide all required inputs."

        # Load models dynamically
        scaler, rf_model, kan_model = load_models(mode)

        # Preprocess input data
        predict_features_scaled = scaler.transform(predict_features)
        predict_features_imputed = SimpleImputer(strategy='mean').fit_transform(predict_features_scaled)

        # Predictions
        rf_predictions = rf_model.predict(predict_features_imputed)
        kan_predictions = kan_model.predict(predict_features_imputed)

        # Plot results
        plt.figure(figsize=(10, 6))
        scale_kan, shape_kan = np.log(kan_predictions[:, 0]), np.sqrt(np.log(1 + (kan_predictions[:, 1] / kan_predictions[:, 0]) ** 2))
        scale_rf, shape_rf = np.log(rf_predictions[:, 0]), np.sqrt(np.log(1 + (rf_predictions[:, 1] / rf_predictions[:, 0]) ** 2))
        x = np.linspace(0.01, 10, 500)

        plt.plot(x, lognorm.pdf(x, s=shape_kan, scale=np.exp(scale_kan[0])), 'r-', label='KAN')
        plt.plot(x, lognorm.pdf(x, s=shape_rf, scale=np.exp(scale_rf[0])), 'g--', label='RF')
        plt.xlabel("Distance (m)")
        plt.ylabel("PDF")
        plt.legend()
        plot_filename = f'static/plot_{mode}.png'
        plt.savefig(plot_filename)
        plt.close()

        return render_template("index.html", plot_url=url_for('static', filename=plot_filename),
                               calculated_values=calculated_values, mode=mode, velocity=velocity,
                               water_depth=water_depth, diameter=diameter)

    return render_template("index.html", plot_url=None, mode=mode)


if __name__ == "__main__":
    app.run(debug=True)
