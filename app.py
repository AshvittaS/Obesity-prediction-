import gradio as gr
import pickle
import numpy as np
import pandas as pd

with open("power_transformer.pkl", "rb") as f:
    pt = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)   # Your trained model file

numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
            'SMOKE', 'SCC', 'CALC', 'MTRANS']

label_mapping = {
    0: "Obesity_Type_II",
    1: "Normal_Weight",
    2: "Overweight"
}

def predict_obesity(Gender, Age, Height, Weight, family_history_with_overweight,
                    FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC,
                    FAF, TUE, CALC, MTRANS):
    
    input_data = pd.DataFrame([[Gender, Age, Height, Weight, family_history_with_overweight,
                                FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC,
                                FAF, TUE, CALC, MTRANS]],
                              columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                                       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
                                       'FAF', 'TUE', 'CALC', 'MTRANS'])
 
    for col in cat_cols:
        le = encoders[col]
        input_data[col] = le.transform(input_data[col])

    input_data[numeric_cols] = pt.transform(input_data[numeric_cols])
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    pred = model.predict(input_data)[0]
    return label_mapping.get(pred, "Unknown")

demo = gr.Interface(
    fn=predict_obesity,
    inputs=[
        gr.Dropdown(["Female", "Male"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Height (m)"),
        gr.Number(label="Weight (kg)"),
        gr.Dropdown(["yes", "no"], label="Family History with Overweight"),
        gr.Dropdown(["yes", "no"], label="FAVC (Frequent High Caloric Food)"),
        gr.Number(label="FCVC (Vegetable Consumption)"),
        gr.Number(label="NCP (Meals per Day)"),
        gr.Dropdown(["Sometimes", "Frequently", "Always", "no"], label="CAEC (Eating Between Meals)"),
        gr.Dropdown(["yes", "no"], label="SMOKE"),
        gr.Number(label="CH2O (Water Consumption per Day)"),
        gr.Dropdown(["yes", "no"], label="SCC (Monitor Caloric Intake)"),
        gr.Number(label="FAF (Physical Activity Frequency)"),
        gr.Number(label="TUE (Time Using Technology per Day)"),
        gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], label="CALC (Alcohol Consumption)"),
        gr.Dropdown(["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"], label="MTRANS")
    ],
    outputs=gr.Textbox(label="Predicted Obesity Level"),
    title="Obesity Level Prediction App",
    description="Predicts obesity category based on lifestyle and health attributes."
)

if __name__ == "__main__":
    demo.launch()
