# Airbnb Toronto Price Prediction
The AI-driven price prediction model aims to predict the prices of Airbnb listings in Toronto using three models: Ridge Regression, Light GBM, and Multi-layered Perceptron (Feed-forward Neural Network). The application is built using Streamlit and various data science libraries.

## Table of Contents
- [Process](#process)
- [Models Evaluation](#models-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Process
#### Data Ingestion
<li> Dataset containing listings for Toronto available via Inside Airbnb. The dataset contains 20k+ listings with 74 different characteristics.</li>

#### Initial Feature Elimination 
<li> Removed any unnecessary columns, renamed columns for better readability and converted price column to float. </li>

#### Exploratory Data Analysis (EDA)
<li> Explored various features of data to identify patterns and visualized using different graphs.</li>

#### Data Preprocessing
<li> Train-test split the data by 90-10 percentages </li>
<li> Took natural log of price </li>
<li> Handled missing values through imputation and removal depending on requirements. </li>
<li> Engineered new features such as days_since columns for host registration</li>
<li> Engineered new features for distance to downtown</li>
<li> Encoded categorical columns using boolean, ordinal and one-hot encoding depending on requirements </li>

#### Modeling and Evaluation
<li> Built 3 different models. </li>
<li>Ridge Regression with regularization </li>
<li> LightGBM with correct parameters to prevent overfitting </li>
<li> Multi-layered Perceptron (MLP) with 3 hidden layers </li>
<li> Evaluated using MSE and R2 scores </li>

#### Model Interpretation
<li> Used SHAP for local and global interpretations </li>

## Models Evaluation
![models_eval](https://github.com/user-attachments/assets/bc294a5c-7c52-4d58-b118-5efcb542f52d)


## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/airbnb-dynamic-pricing-toronto.git
    cd airbnb-dynamic-pricing-toronto
    ```

2. **Create a virtual environment:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

2. **Access the application:**

    Open your web browser and go to `http://localhost:8501`.

3. **Online Access:**

    You can also access the application online at [this link](https://airbnbdyanmicpricingoftoronto-vmbeauzhxexrihabelyca2.streamlit.app/).

## Project Structure

- `app.py`: Main application file.
- `requirements.txt`: List of required Python packages.
- `dataset/`: Directory containing Airbnb listing dataset.
- `backend/`: Jupyter notebooks for data exploration and model training

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
