import zipfile

with zipfile.ZipFile(
    "models/cough-classification-lstm/INPUT_model_path/lstm_model.keras"
) as z:
    print(z.namelist())