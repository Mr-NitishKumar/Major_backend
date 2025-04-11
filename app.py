from flask import Flask, request, jsonify, session
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from dataset_utils import analyze_dataset, categorical_summary, correlation_heatmap, descriptive_statistics, encode_categorical_data, generate_histograms, handle_missing_values, missing_value_summary, preprocess_dataset, scale_features, split_dataset
from prediction_utils import evaluate_classification, evaluate_regression, predict_model
from visualization_utils import create_box_plots, create_clustering_plot, create_interactive_scatter, create_pair_plot
import os
from flask_cors import CORS
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
CORS(app)

app.secret_key = 'your_secret_key'  # For session handling

DB_NAME = 'users.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                dob TEXT NOT NULL,
                gender TEXT NOT NULL,
                address TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()




# Endpoint to upload file once and store it in the session
@app.route('/upload-once', methods=['POST'])
def upload_once():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    # Save the file temporarily
    session['uploaded_file_path'] = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    file.save(session['uploaded_file_path'])

    # Read the dataset and return column names (optional)
    df = pd.read_csv(session['uploaded_file_path'])
    feature_names = df.columns.tolist()
    return jsonify({"message": "File uploaded successfully!", "features": feature_names})


# --- LOGIN ROUTE ---
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password required"}), 400

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Fetch user by username
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()

    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    hashed_password = user[0]

    if not check_password_hash(hashed_password, password):
        return jsonify({"success": False, "message": "Incorrect password"}), 401
    
    session['username'] = username

    return jsonify({"success": True, "message": "Login successful"})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    full_name = data.get('fullName')
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirmPassword')
    dob = data.get('dob')
    gender = data.get('gender')
    address = data.get('address')

    if not all([full_name, username, email, password, confirm_password, dob, gender, address]):
        return jsonify({"success": False, "message": "All fields are required."}), 400

    if password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match."}), 400

    hashed_password = generate_password_hash(password)

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (full_name, username,  email, password, dob, gender, address)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (full_name, username, email, hashed_password, dob, gender, address))
            conn.commit()
        return jsonify({"success": True, "message": "User registered successfully."})
    except sqlite3.IntegrityError as e:
        error_msg = str(e)
        if "username" in error_msg:
            return jsonify({"success": False, "message": "Username already exists."}), 409
        elif "email" in error_msg:
            return jsonify({"success": False, "message": "Email already exists."}), 409
        else:
            return jsonify({"success": False, "message": "Database error."}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})


@app.route('/upload', methods=['POST'])
def upload_data():
    file = request.files.get('file')
    if not file or file.filename.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type! Only CSV files are allowed."}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    basic_info = analyze_dataset(file_path)
    return jsonify({"message": "File uploaded successfully!", "basic_info": basic_info})



@app.route('/preprocess', methods=['POST'])
def preprocess():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    # Read the dataset
    df = pd.read_csv(file)

    # Get user-defined preprocessing options
    missing_strategy = request.form.get('missing_strategy', 'mean')
    scaling_method = request.form.get('scaling_method', 'standard')
    encoding_method = request.form.get('encoding_method', 'onehot')
    test_size = float(request.form.get('test_size', 0.2))
    target_column = request.form.get('target')

    # Apply preprocessing steps
    df = handle_missing_values(df, strategy=missing_strategy)
    df = scale_features(df, method=scaling_method)
    df = encode_categorical_data(df, encoding=encoding_method)

    if target_column:
        X_train, X_test, y_train, y_test = split_dataset(df, target_column, test_size=test_size)
        return jsonify({
            "message": "Preprocessing completed!",
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape
        })
    
    return jsonify({"message": "Preprocessing completed!", "dataset_shape": df.shape})


@app.route('/eda', methods=['POST'])
def eda():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    # Read the dataset
    df = pd.read_csv(file)

    # Perform EDA
    stats = descriptive_statistics(df)
    missing_values = missing_value_summary(df)
    categorical_data = categorical_summary(df)
    histograms = generate_histograms(df)
    heatmap = correlation_heatmap(df)

    return jsonify({
        "stats": stats,
        "missing_values": missing_values,
        "categorical_data": categorical_data,
        "histograms": histograms,
        "correlation_heatmap": heatmap,
    })

@app.route('/eda/features', methods=['POST'])
def upload_and_get_features():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    # Read the dataset
    df = pd.read_csv(file)

    # Extract feature names (column names)
    feature_names = df.columns.tolist()
    return jsonify({"features": feature_names})

@app.route('/eda/analyze', methods=['POST'])
def perform_eda():
    file = request.files.get('file')
    selected_features = request.form.get('selected_features').split(',')

    # Read the dataset
    df = pd.read_csv(file)

    # Filter dataset to selected features
    df_selected = df[selected_features]

    # Perform EDA (example: descriptive statistics)
    eda_results = df_selected.describe().to_dict()
    return jsonify({"eda_results": eda_results})

@app.route('/clean-data', methods=['POST'])
def clean_data():
    file_path = session.get('uploaded_file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No file found in session!"}), 400
    
    # Read dataset
    df = pd.read_csv(file_path)

    # Get user-selected cleaning options
    drop_missing = request.form.get('drop_missing')  # 'rows' or 'columns'
    fill_missing = request.form.get('fill_missing')  # 'mean', 'median', 'mode'
    remove_duplicates = request.form.get('remove_duplicates')  # 'true' or 'false'

    # Apply missing value handling
    if drop_missing == 'rows':
        df = df.dropna()
    elif drop_missing == 'columns':
        df = df.dropna(axis=1)
    elif fill_missing == 'mean':
        df = df.fillna(df.mean())
    elif fill_missing == 'median':
        df = df.fillna(df.median())
    elif fill_missing == 'mode':
        df = df.fillna(df.mode().iloc[0])

    # Remove duplicates
    if remove_duplicates == 'true':
        df = df.drop_duplicates()

    # Return cleaned data feature names
    feature_names = df.columns.tolist()
    return jsonify({"cleaned_features": feature_names})



@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    model_type = request.form.get('model')
    features = request.form.getlist('features')
    target = request.form.get('target')
    params = request.form.to_dict()

    df = pd.read_csv(file)
    X = df[features]
    y = df[target] if target in df.columns else None

    predictions = predict_model(model_type, X, y, params)
    return jsonify({"predictions": predictions.tolist()})

@app.route('/visualize', methods=['POST'])
def visualize():
    file = request.files.get('file')
    n_clusters = int(request.form.get('n_clusters', 3))

    df = pd.read_csv(file)
    numeric_data = df.select_dtypes(include=['number'])
    clusters = predict_model('kmeans', numeric_data, params={"n_clusters": n_clusters})

    plot_path = create_clustering_plot(numeric_data, clusters)
    return jsonify({"plot_url": f"/{plot_path}"})

@app.route('/visualize/advanced', methods=['POST'])
def advanced_visualize():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    # Read the dataset
    df = pd.read_csv(file)

    # Get visualization type from user
    vis_type = request.form.get('vis_type')
    x_col = request.form.get('x_col')
    y_col = request.form.get('y_col')

    if vis_type == "pair_plot":
        pair_plot = create_pair_plot(df)
        return jsonify({"pair_plot": pair_plot})

    elif vis_type == "box_plot":
        box_plots = create_box_plots(df)
        return jsonify({"box_plots": box_plots})

    elif vis_type == "scatter" and x_col and y_col:
        scatter_plot = create_interactive_scatter(df, x_col, y_col)
        return jsonify({"scatter_plot": scatter_plot})

    return jsonify({"error": "Invalid visualization type or parameters!"}), 400

@app.route('/evaluate', methods=['POST'])
def evaluate():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    # Read the dataset
    df = pd.read_csv(file)
    features = request.form.getlist('features')
    target = request.form.get('target')
    model_type = request.form.get('model')

    # Train and predict
    X = df[features]
    y = df[target]
    predictions = predict_model(model_type, X, y)

    # Choose evaluation metrics based on model type
    if model_type in ['linear', 'random_forest', 'svm']:  # Regression
        metrics = evaluate_regression(y, predictions)
    elif model_type in ['logistic_regression']:  # Classification
        metrics = evaluate_classification(y, predictions)
    else:
        return jsonify({"error": "Invalid model type for evaluation!"}), 400

    return jsonify({"metrics": metrics})







if __name__ == '__main__':
    app.run(debug=True)
