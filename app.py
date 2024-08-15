
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import  Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
# %pip install shap
import shap
import streamlit as st
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
warnings.filterwarnings('ignore')

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def ensure_nltk_resources():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

# Ensure NLTK resources are available
ensure_nltk_resources()

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

airbnb_df = pd.read_csv('dataset/Airbnb/Toronto.csv')
airbnb_df['price'] = airbnb_df['price'].str.replace('[$,]', '', regex=True).astype(float)

with st.container():
    st.header("Airbnb listing dataset overview")
    st.write(airbnb_df.head(10))


with st.container():
    st.header("Visualizations")

    # Create a figure with two subplots

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot Room Type Distribution

    room_type_counts = airbnb_df['room_type'].value_counts()

    sns.barplot(x=room_type_counts.index, y=room_type_counts.values, ax=ax[0])

    ax[0].set_title('Room Type Distribution')

    ax[0].set_xlabel('Room Type')

    ax[0].set_ylabel('Count')

    # Plot Average Price per Room Type

    avg_price_per_room = airbnb_df.groupby('room_type')['price'].mean().reset_index()

    sns.barplot(x='room_type', y='price', data=avg_price_per_room, ax=ax[1])

    ax[1].set_title('Average Price per Room Type')

    ax[1].set_xlabel('Room Type')

    ax[1].set_ylabel('Average Price')

    # Show the plots

    plt.tight_layout()

    plt.show()

    st.pyplot(fig)

_, col2, _ = st.columns(3)
with col2:
    run_pred_model = st.button("Train the Models")
#Add loading bar for the displaying the model evaluation metrices

measures_values_are_available = False
if run_pred_model:
    
    feature_elimination_list = ['amenities', 'host_verifications', 'listing_url', 'attractions_within_25km', 'City',
                                'scrape_id', 'last_scraped', 'source', 'host_url', 'host_thumbnail_url',
                                'calendar_updated', 'calendar_last_scraped', 'host_name', 'id', 'host_id',
                                'neighbourhood_group_cleansed', 'host_neighbourhood', 'host_location', 'neighbourhood',
                                'neighbourhood_cleansed', 'host_listings_count', 'host_total_listings_count',
                                'calculated_host_listings_count_entire_homes',
                                'calculated_host_listings_count_private_rooms',
                                'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
                                'review_scores_accuracy', 'bathrooms_text', 'minimum_minimum_nights',
                                'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
                                'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'number_of_reviews_ltm',
                                'number_of_reviews_l30d', 'host_about', 'neighborhood_overview', 'host_has_profile_pic',
                                'picture_url', 'host_picture_url', 'calculated_host_listings_count',
                                'number_of_reviews']

    airbnb_df = airbnb_df.drop(feature_elimination_list, axis=1)

    column_mappings = {
        'name': 'title',
        'first_review': 'first_review_date',
        'last_review': 'last_review_date',
        'review_scores_value': 'review_scores_value_for_money'
    }

    # Renaming columns
    airbnb_df.rename(columns=column_mappings, inplace=True)

    # Removing "$" from price and converting to float
    # print(airbnb_df['price'].dtype)


    #train-test split 
    data_train, data_test = train_test_split(airbnb_df, test_size=0.10, random_state=42)

    data_train['price'] = np.log(data_train['price'])
    non_numeric_columns = data_train.select_dtypes(include=['object']).columns
    data_train_numeric = data_train.drop(non_numeric_columns, axis=1)

    # Fill null value with unlicensed
    data_train['license'].fillna('Unlicensed', inplace=True)
    
    # Drop rows with empty price
    data_train = data_train.dropna(subset=['price'])
    data_train = data_train.dropna(subset=['review_scores_rating'])
    data_train['host_is_superhost'] = data_train['host_is_superhost'].fillna('f')
    data_train['has_availability'] = data_train['has_availability'].fillna('f')
    data_train['host_response_time'] = data_train['host_response_time'].fillna(data_train['host_response_time'].mode()[0])

    # Remove % sign and convert to numeric for the following columns
    data_train['host_response_rate'] = pd.to_numeric(data_train['host_response_rate'].str.replace('%', ''))
    data_train['host_acceptance_rate'] = pd.to_numeric(data_train['host_acceptance_rate'].str.replace('%', ''))

    # Creating a list of required columns
    numeric_columns = [
        'host_response_rate',
        'host_acceptance_rate',
        'bedrooms', 'beds', 
        'review_scores_value_for_money', 
        'review_scores_location', 
        'review_scores_checkin', 
        'review_scores_communication', 
        'review_scores_cleanliness', 
        'bathrooms'
    ]

    # Fill null values in specified columns with median values
    for column in numeric_columns:
        median_value = data_train[column].median()
        data_train[column].fillna(median_value, inplace=True)  

    current_date = datetime.now()

    # Converting date columns
    data_train['host_since'] = pd.to_datetime(data_train['host_since'])

    # Calculating values and storing in a new column
    data_train['host_since_days'] = (current_date - data_train['host_since']).dt.days

    # Dropping date columns
    data_train.drop(columns=['host_since', 'first_review_date', 'last_review_date'], inplace=True)

    # Define downtown coordinates for Toronto city
    downtown_coords = (43.6532, -79.3832) 

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    #unpack the coordinates
    lat,lon = downtown_coords 

    # Calculate the distance to downtown for each entry in the dataset
    data_train['distance_to_downtown'] = data_train.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], lat, lon), axis=1
    )

    # Selecting non-numerical columns from the dataframe
    non_numerical_columns = data_train.select_dtypes(exclude=['number']).columns.tolist()
    categorical_columns = data_train[non_numerical_columns]

    categorical_columns.head()
    for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        data_train[boolean_column] = data_train[boolean_column].map(lambda s: False if s == "f" else True)
    
    # Changing data
    data_train['license'] = data_train['license'].map(lambda s: False if s == "Unlicensed" else True)

    # Define the order of categories
    categories = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']

    # Initialize OrdinalEncoder with the defined categories
    ordinal_encoder = OrdinalEncoder(categories=[categories])

    # Fit and transform the 'host_response_time' column
    data_train['host_response_time_encoded'] = ordinal_encoder.fit_transform(data_train[['host_response_time']])

    # Dropping the categorical column
    data_train.drop(columns=['host_response_time'], inplace=True)

    # Create the dummy variables without dropping the original column
    dummy_vars_tp = pd.get_dummies(data_train['property_type'], prefix='property')
    st.session_state.dummy_vars_tp = dummy_vars_tp
    # Concatenate the original DataFrame with the dummy variables
    data_train = pd.concat([data_train, dummy_vars_tp], axis=1)

    # Create the dummy variables without dropping the original column
    dummy_vars_tr = pd.get_dummies(data_train['room_type'], prefix='room_type')
    st.session_state.dummy_vars_tr = dummy_vars_tr

    # Concatenate the original DataFrame with the dummy variables
    data_train = pd.concat([data_train, dummy_vars_tr], axis=1)


    #creating an object of sentiment intensity analyzer
    # sia= SentimentIntensityAnalyzer()

    # creating new columns using polarity scores function
    data_train['title_scores'] = data_train['title'].apply(lambda title: sia.polarity_scores(str(title)))
    data_train['title_sentiment']=data_train['title_scores'].apply(lambda score_dict:score_dict['compound'])
    data_train.drop(['title', 'title_scores'], axis=1, inplace=True)


    # creating new columns using polarity scores function
    data_train['description_scores']=data_train['description'].apply(lambda description: sia.polarity_scores(str(description)))
    data_train['description_sentiment']=data_train['description_scores'].apply(lambda score_dict:score_dict['compound'])
    data_train.drop(['description', 'description_scores'], axis=1, inplace=True)

    train_features = data_train.copy()

    data_test['price'] = np.log(data_test['price'])
    non_numeric_columns = data_test.select_dtypes(include=['object']).columns
    data_test_numeric = data_test.drop(non_numeric_columns, axis=1)

    # Fill null value with unlicensed
    data_test['license'].fillna('Unlicensed', inplace=True)
    
    # Drop rows with empty price
    data_test = data_test.dropna(subset=['price'])
    data_test = data_test.dropna(subset=['review_scores_rating'])
    data_test['host_is_superhost'] = data_test['host_is_superhost'].fillna('f')
    data_test['has_availability'] = data_test['has_availability'].fillna('f')
    data_test['host_response_time'] = data_test['host_response_time'].fillna(data_test['host_response_time'].mode()[0])

    # Remove % sign and convert to numeric for the following columns
    data_test['host_response_rate'] = pd.to_numeric(data_test['host_response_rate'].str.replace('%', ''))
    data_test['host_acceptance_rate'] = pd.to_numeric(data_test['host_acceptance_rate'].str.replace('%', ''))

    # Creating a list of required columns
    numeric_columns = [
        'host_response_rate',
        'host_acceptance_rate',
        'bedrooms', 'beds', 
        'review_scores_value_for_money', 
        'review_scores_location', 
        'review_scores_checkin', 
        'review_scores_communication', 
        'review_scores_cleanliness', 
        'bathrooms'
    ]

    # Fill null values in specified columns with median values
    for column in numeric_columns:
        median_value = data_test[column].median()
        data_test[column].fillna(median_value, inplace=True)  

    current_date = datetime.now()

    # Converting date columns
    data_test['host_since'] = pd.to_datetime(data_test['host_since'])

    # Calculating values and storing in a new column
    data_test['host_since_days'] = (current_date - data_test['host_since']).dt.days

    # Dropping date columns
    data_test.drop(columns=['host_since', 'first_review_date', 'last_review_date'], inplace=True)

    
    

    # Calculate the distance to downtown for each entry in the dataset
    data_test['distance_to_downtown'] = data_test.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], lat, lon), axis=1
    )

    # Selecting non numerical columns from the dataframe
    non_numerical_columns = data_test.select_dtypes(exclude=['number']).columns.tolist()
    categorical_columns = data_test[non_numerical_columns]

    categorical_columns.head()
    for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        data_test[boolean_column] = data_test[boolean_column].map(lambda s: False if s == "f" else True)
    
    # Changing data
    data_test['license'] = data_test['license'].map(lambda s: False if s == "Unlicensed" else True)

    # Define the order of categories
    categories = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']

    # Initialize OrdinalEncoder with the defined categories
    ordinal_encoder = OrdinalEncoder(categories=[categories])

    # Fit and transform the 'host_response_time' column
    data_test['host_response_time_encoded'] = ordinal_encoder.fit_transform(data_test[['host_response_time']])

    # Dropping the categorical column
    data_test.drop(columns=['host_response_time'], inplace=True)

    # Create the dummy variables without dropping the original column
    dummy_test_vars = pd.get_dummies(data_test['property_type'], prefix='property')

    # Align the columns in data_test to match those in data_train
    dummy_test_vars = dummy_test_vars.reindex(columns=dummy_vars_tp.columns, fill_value=0)

    # Concatenate the original DataFrame with the dummy variables
    data_test = pd.concat([data_test, dummy_test_vars], axis=1)

    # Create the dummy variables without dropping the original column
    dummy_test_vars = pd.get_dummies(data_test['room_type'], prefix='room_type')

    # Align the columns in data_test to match those in data_train
    dummy_test_vars = dummy_test_vars.reindex(columns=dummy_vars_tr.columns, fill_value=0)

    # Concatenate the original DataFrame with the dummy variables
    data_test = pd.concat([data_test, dummy_test_vars], axis=1)


    #creating an object of sentiment intensity analyzer
    # sia= SentimentIntensityAnalyzer()

    # creating new columns using polarity scores function
    data_test['title_scores'] = data_test['title'].apply(lambda title: sia.polarity_scores(str(title)))
    data_test['title_sentiment']=data_test['title_scores'].apply(lambda score_dict:score_dict['compound'])
    data_test.drop(['title', 'title_scores'], axis=1, inplace=True)


    # creating new columns using polarity scores function
    data_test['description_scores']=data_test['description'].apply(lambda description: sia.polarity_scores(str(description)))
    data_test['description_sentiment']=data_test['description_scores'].apply(lambda score_dict:score_dict['compound'])
    data_test.drop(['description', 'description_scores'], axis=1, inplace=True)

    test_features = data_test.copy()

        #Handling Outliers using median and IQR 
    def outlier(a):
        Q1 = a.quantile(0.25)
        Q3 = a.quantile(0.75)
        IQR = Q3-Q1
        L = Q1 - 1.5*IQR
        U = Q3 + 1.5*IQR
        return(L,U)

    # # Updating price
    train_features = train_features.loc[train_features['price'] >= outlier(train_features['price'])[0]]
    train_features = train_features.loc[train_features['price'] <= outlier(train_features['price'])[1]]

    test_features = test_features.loc[test_features['price'] >= outlier(test_features['price'])[0]]
    test_features = test_features.loc[test_features['price'] <= outlier(test_features['price'])[1]]


    # st.write(train_features)


    # X_train_selected.shape, X_test_selected.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # Feature Scaling
    scaler = StandardScaler()
    X_train = train_features.drop(columns=['price'])
    X_test = test_features.drop(columns=['price'])


    #Drop property_type and room_type columns
    X_train.drop(columns=['property_type', 'room_type'],inplace=True)

    # Aligning column discrepancy due to separating property types. Important step.
    common_columns = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    # Ridge Regression
    model_ridge = Ridge(alpha=0.13, random_state=42) # Used RidgeCV to identify ideal hyperparameter
    y_train = train_features['price']
    model_ridge.fit(X_train_scaled, y_train)

    # Predicting the price
    train_pred_ridge = model_ridge.predict(X_train_scaled)
    test_pred_ridge = model_ridge.predict(X_test_scaled)

    # Evaluate Ridge Regression
    r_train_r2 = r2_score(y_train, train_pred_ridge)
    r_train_mse = mean_squared_error(y_train, train_pred_ridge)
    y_test = test_features['price']
    r_test_r2 = r2_score(y_test, test_pred_ridge)
    r_test_mse = mean_squared_error(y_test, test_pred_ridge)

    # Printing the evaluation metrices values
    print(f'\n--------Ridge Regression Train Fitting-----------')
    print(f'R2 Score: {r_train_r2}')
    print(f'MSE: {r_train_mse}')
    print(f'--------Ridge Regression Test Fitting-----------')
    print(f'R2 Score: {r_test_r2}')
    print(f'MSE: {r_test_mse}')

    # Define the MLP model
    model_mlp = Sequential()

    # Input Layer and First Hidden Layer
    model_mlp.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model_mlp.add(Dropout(0.3))

    # Second Hidden Layer
    model_mlp.add(Dense(64, activation='relu'))

    # Third Hidden Layer
    model_mlp.add(Dense(32, activation='relu'))

    # Output Layer
    model_mlp.add(Dense(1, activation='linear'))

    # Compile the model
    model_mlp.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Implement early stopping and learning rate reduction on plateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train the model
    model_mlp.fit(X_train_scaled, y_train,
                            validation_data=(X_test_scaled, y_test),
                            epochs=100,
                            batch_size=32,
                            callbacks=[early_stopping, reduce_lr],
                            verbose=-1)

    # Predicting the price
    train_pred_mlp = model_mlp.predict(X_train_scaled)
    test_pred_mlp = model_mlp.predict(X_test_scaled)

    # Evaluate MLP
    m_train_r2 = r2_score(y_train, train_pred_mlp)
    m_train_mse = mean_squared_error(y_train, train_pred_mlp)
    m_test_r2 = r2_score(y_test, test_pred_mlp)
    m_test_mse = mean_squared_error(y_test, test_pred_mlp)

    # Calculating and printing the evaluation metrices values
    print(f'\n--------MLP Train Fitting-----------')
    print(f'R2 Score: {m_train_r2}')
    print(f'MSE: {m_train_mse}')

    print(f'--------MLP Test Fitting-----------')
    print(f'R2 Score: {m_test_r2}')
    print(f'MSE: {m_test_mse}')
    measures_values_are_available = True

    #LightGBM
    # Drop encoded columns for LightGBM
    # Identify columns to drop
    cols_to_drop = train_features.filter(regex='^(property_(?!type$)|room_type_)').columns

    # Drop the identified columns
    # These are our training features for the final model
    X_train_selected = train_features.drop(columns=cols_to_drop)
    X_train_selected = X_train_selected.drop(['price'], axis=1)
    X_train_selected = X_train_selected[sorted(X_train_selected.columns)]
    # Identify columns to drop
    # THese are our testing features for the final model
    cols_to_drop = test_features.filter(regex='^(property_(?!type$)|room_type_)').columns
    X_test_selected = test_features.drop(columns=cols_to_drop)
    X_test_selected = X_test_selected.drop(['price'], axis=1)
    X_test_selected = X_test_selected[sorted(X_test_selected.columns)]


    # Drop categorical columns for other models
    X_train = train_features.drop(columns=['price', 'property_type', 'room_type'])
    X_test = test_features.drop(columns=['price', 'property_type', 'room_type'])

    X_train = X_train[sorted(X_train.columns)]
    X_test = X_test[sorted(X_test.columns)]




    # Aligning column discrepancy due to separating property types. Important step.
    st.session_state.X_train_for_model = X_train.columns
    st.session_state.X_test_for_model_data = X_test
    common_columns = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_columns]
    st.session_state.X_train = X_train
    X_test = X_test[common_columns]

    # Selecting target for train and test
    y_train = train_features['price']
    y_test = test_features['price']


    # Build the model

    model_lgbm = lgb.LGBMRegressor(max_bin=100, learning_rate=0.01, n_estimators=1000, num_leaves=50, verbose=0,
                                max_depth=-1, random_state=42)
    # Encode categorical features
    categorical_features = ['property_type', 'room_type']
    for col in categorical_features:
        X_train_selected[col] = X_train_selected[col].astype('category')
        X_test_selected[col] = X_test_selected[col].astype('category')

    # Convert categorical features to their indices
    categorical_feature_indices = [X_train_selected.columns.get_loc(col) for col in categorical_features]

    model_lgbm.fit(X_train_selected, y_train, categorical_feature=categorical_feature_indices)
    train_pred_lgbm = model_lgbm.predict(X_train_selected)
    test_pred_lgbm = model_lgbm.predict(X_test_selected)

    # Evaluate Light GBM
    lgbm_train_r2 = r2_score(y_train, train_pred_lgbm)
    lgbm_train_mse = mean_squared_error(y_train, train_pred_lgbm)
    lgbm_test_r2 = r2_score(y_test, test_pred_lgbm)
    lgbm_test_mse = mean_squared_error(y_test, test_pred_lgbm)

    # Calculating and printing the evaluation metrices values
    print(f'\n--------Tuned Light GBM Train Fitting-----------')
    print(f'R2 Score: {lgbm_train_r2}')
    print(f'MSE: {lgbm_train_mse}')
    print(f'--------Tuned Light GBM Test Fitting-----------')
    print(f'R2 Score: {lgbm_test_r2}')
    print(f'MSE: {lgbm_test_mse}')

    r2_difference = lgbm_train_r2 - lgbm_test_r2
    mse_difference = lgbm_train_mse - lgbm_test_mse
    print(f'Difference in R2 Score: {r2_difference}')
    print(f'Difference in MSE: {mse_difference}')

   

    # how to save the model in the session state
    saved_model = {
        'ridge': model_ridge,
        'lightgbm': model_lgbm,
        'mlp': model_mlp
    }


    st.session_state.model = saved_model

    saved_model_evaluation = {
        'ridge': [r_train_r2, r_train_mse, r_test_r2, r_test_mse],
        'lightgbm': [lgbm_train_r2, lgbm_train_mse, lgbm_test_r2, lgbm_test_mse],
        'mlp': [m_train_r2, m_train_mse, m_test_r2, m_test_mse]
    }

    st.session_state.model_evaluation = saved_model_evaluation

    # st.write(st.session_state.model)


#run the model evaluation only if the st.session_state.model is not None
if 'model'  in  st.session_state:
    with st.container():
        st.title('Model Evaluation')
        # Create a DataFrame for easy plotting
        results_df = pd.DataFrame({
            'Model': ['Ridge', 'LightGBM', 'MLP'],
            #Load from the session state model evaluation
            'R² Score': [st.session_state.model_evaluation['ridge'][2], st.session_state.model_evaluation['lightgbm'][2], st.session_state.model_evaluation['mlp'][2]],
            # 'R² Score': [r_test_r2, lgbm_test_r2, m_test_r2],
            #Load from the session state model evaluation
            'MSE': [st.session_state.model_evaluation['ridge'][3], st.session_state.model_evaluation['lightgbm'][3], st.session_state.model_evaluation['mlp'][3]]
            # 'MSE': [r_test_mse, lgbm_test_mse, m_test_mse]
        })

        # Find the index of the highest R² score and lowest MSE
        best_r2_idx = results_df['R² Score'].idxmax()
        best_mse_idx = results_df['MSE'].idxmin()

        # Set up the subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot R² scores with custom colors
        palette_r2 = ['red' if i == best_r2_idx else 'blue' for i in range(len(results_df))]
        sns.barplot(ax=axes[0], x='Model', y='R² Score', data=results_df, palette=palette_r2)
        axes[0].set_title('R² Score Comparison of Regression Models')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('R² Score')

        # Plot MSE with custom colors
        palette_mse = ['red' if i == best_mse_idx else 'blue' for i in range(len(results_df))]
        sns.barplot(ax=axes[1], x='Model', y='MSE', data=results_df, palette=palette_mse)
        axes[1].set_title('Mean Squared Error Comparison of Regression Models')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Mean Squared Error')

        plt.tight_layout()

        # Display the plots in Streamlit
        st.pyplot(plt.gcf())

    
        col1, col2 = st.columns(2)
        with col1:
            # Display the DataFrame in Streamlit
            st.write("R² Scores:")
            st.dataframe(results_df[['Model', 'R² Score']])
        with col2:
            st.write("\nMean Squared Errors:")
            st.dataframe(results_df[['Model', 'MSE']])


st.markdown('---')


# Set up header and brief description
with st.container():
    st.header('Airbnb Price Predictor for Tornoto city')
    st.markdown('Provide data about your Airbnb listing and get predictions!')

# Begin new section for listings features
st.markdown('---')
st.subheader('Airbnb Listing characteristics')



# Streamlit UI setup
st.subheader("Toronto Address Locator with Map")
# Get user address input
# address = st.text_input("Enter your address:")
# Geocode address to get initial location

# import streamlit as st
# import folium
# from folium import IFrame
# from streamlit_folium import st_folium
# from folium.plugins import MiniMap
#
# # Function to check if coordinates are within Toronto
# def is_within_toronto(lat, lon):
#     return (43.6 <= lat <= 43.85) and (-79.5 <= lon <= -79.0)
#
# # Streamlit UI
# st.title('Interactive Map with Pin')
#
# # Create a Folium map centered on Toronto
# toronto_map = folium.Map(location=[43.7, -79.42], zoom_start=12)
#
# # Add a minimap
# minimap = MiniMap()
# toronto_map.add
#
# location = geolocator.geocode(address)
# if location and is_within_toronto(location.latitude, location.longitude):
#     st.success(f"Location found: {location.address}")
#     lat, lon = location.latitude, location.longitude
# else:
#     st.error("Address not found or not within Toronto. Please try another address.")
#     lat, lon = 43.7, -79.4  # Default to Toronto center if out of bounds
#
#
#  # Create the Folium map centered on the location
# m = folium.Map(location=[lat, lon], zoom_start=12)
#
#
# # Add a draggable marker
# marker = folium.Marker(
#     location=[lat, lon],
#     draggable=True,
#     popup="Move me!"
#
# )
#
# marker.add_to(m)
#
#
#
# # Add MousePosition plugin to get lat/lon on click
# MousePosition().add_to(m)
#
#
# # Capture the new location when marker is moved
# map_data = st_folium(m, width=700, height=500)
#
#
# # Display latitude and longitude of the marker
# if map_data["last_object_clicked"]:
#     clicked_lat = map_data["last_object_clicked"]["lat"]
#     clicked_lon = map_data["last_object_clicked"]["lng"]
#
#     if is_within_toronto(clicked_lat, clicked_lon):
#         st.write(f"Selected location - Latitude: {clicked_lat}, Longitude: {clicked_lon}")
#     else:
#         st.warning("Selected location is outside Toronto boundaries. Please select within Toronto.")

import streamlit as st
import folium
from streamlit_folium import st_folium

# Toronto bounds
TORONTO_BOUNDS = {
    'lat_min': 43.6,
    'lat_max': 43.85,
    'lon_min': -79.5,
    'lon_max': -79.0
}


def is_within_bounds(lat, lon):
    return (TORONTO_BOUNDS['lat_min'] <= lat <= TORONTO_BOUNDS['lat_max']) and (
                TORONTO_BOUNDS['lon_min'] <= lon <= TORONTO_BOUNDS['lon_max'])


# Center of Toronto
center_lat = 43.7
center_lon = -79.42

# Initialize variables to store latitude and longitude
lat = center_lat
lon = center_lon

# Create a Folium map centered on Toronto
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add a draggable marker for the city center
marker = folium.Marker(
    location=[center_lat, center_lon],
    draggable=True,
    popup="Drag me to select a location"
)
marker.add_to(m)

# Display the map
st_data = st_folium(m, width=700, height=500)

# Update lat and lon based on user interaction
if 'last_object_clicked' in st_data and st_data['last_object_clicked']:
    lat = st_data['last_object_clicked']['lat']
    lon = st_data['last_object_clicked']['lng']

    if is_within_bounds(lat, lon):
        lat = st_data['last_object_clicked']['lat']
        lon = st_data['last_object_clicked']['lng']
        # st.write(f"Marker Location: Latitude = {lat}, Longitude = {lon}")
    else:
        # Inform user that the location is outside Toronto
        st.write("Selected location is outside Toronto.")

        lat = center_lat
        lon = center_lon

        # For visual consistency, reset marker to center of Toronto
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        folium.Marker(location=[center_lat, center_lon], draggable=True, popup="Drag me to select a location").add_to(m)
        # st_data = st_folium(m, width=700, height=500)
else:
    st.write("Drag the marker to select a location.")

# Use lat and lon as needed
# For example, you might want to display or process these values elsewhere in your application
st.write(f"Stored Coordinates: Latitude = {lat}, Longitude = {lon}")

listing_title = st.text_input("Enter the list title")
listing_description = st.text_area("Enter the list description")

col1, col2 = st.columns(2)
with col1:
    accommodates = st.slider('Maximum Occupancy', 1, 16, 4)
    bathrooms = st.slider('Number of bathrooms', 1, 9, 2)
    room_type = st.selectbox('Room Type',
                             ('Private room', 'Entire apartment', 'Shared room', 'Hotel room'))
    listing_availability = st.selectbox('Is Listing Available?',('Yes','No'))
    instant_bookable = st.selectbox('Can the listing be instantly booked?',
                           ('No', 'Yes'))
    property_type =  st.selectbox('Select property type',('Entire rental unit', 'Entire home', 'Private room in rental unit',
       'Entire loft', 'Entire condo', 'Entire guest suite',
       'Private room in townhouse', 'Entire cottage',
       'Private room in condo', 'Private room in bed and breakfast',
       'Private room in home', 'Entire serviced apartment',
       'Entire townhouse', 'Private room', 'Room in aparthotel',
       'Private room in serviced apartment', 'Private room in loft',
       'Private room in cottage', 'Entire bungalow',
       'Entire vacation home', 'Shared room in rental unit', 'Tiny home',
       'Shared room in loft', 'Entire villa', 'Private room in bungalow',
       'Private room in guesthouse', 'Entire chalet',
       'Private room in hostel', 'Private room in villa',
       'Room in hostel', 'Entire guesthouse', 'Shared room in hostel',
       'Room in hotel', 'Shared room in home', 'Room in boutique hotel',
       'Private room in guest suite', 'Casa particular',
       'Shared room in condo', 'Private room in vacation home',
       'Private room in minsu', 'Religious building', 'Castle',
       'Entire cabin', 'Shared room in hotel', 'Boat', 'Entire place',
       'Campsite', 'Yurt', 'Camper/RV', 'Entire bed and breakfast',
       'Private room in farm stay', 'Barn', 'Room in bed and breakfast',
       'Private room in castle', 'Treehouse', 'Farm stay', 'Dome',
       'Room in serviced apartment', 'Private room in island',
       'Private room in tent', 'Private room in nature lodge',
       'Private room in earthen home', 'Private room in tiny home',
       'Private room in hut', 'Shared room in townhouse', 'Tent',
       'Island', 'Train', 'Private room in chalet',
       'Private room in casa particular', 'Private room in resort',
       'Shipping container', 'Tower', 'Bus', 'Private room in cabin',
       'Shared room in guesthouse', 'Cave', 'Shared room in tiny home',
       'Private room in cave', 'Entire home/apt', 'Private room in barn',
       'Floor', 'Shared room in bungalow', 'Shared room in guest suite',
       'Earthen home', 'Shared room in boat',
       'Shared room in bed and breakfast', 'Private room in treehouse',
       'Shared room', 'Private room in boat', 'Entire timeshare',
       'Private room in camper/rv', 'Houseboat', 'Private room in dome',
       'Private room in religious building'))

with col2:
    beds = st.slider('Number of beds', 1, 32, 2)
    bedrooms = st.slider('Number of bedrooms', 1, 24, 2)
    min_nights = st.slider('Minimum number of nights', 1, 20, 3)
    max_nights = st.slider('Maximum number of nights', 1, 20, 3)


number_of_days_available_in_a_month = st.slider('Number of days available', 1, 31, 1)
number_of_days_available_in_two_months = st.slider('Number of days available', 2, 60, 1)
number_of_days_available_in_three_months = st.slider('Number of days available', 3, 90, 1)
number_of_days_available_in_year = st.slider('Number of days available', 12, 365, 1)



# Section for host info
st.markdown('---')
st.subheader('Host Information')
col1, col2 = st.columns(2)
with col1:
    host_registration_date = st.date_input("Enter host registration date",datetime.today())
    super_host = st.selectbox('Is your host a superhost?', ('No', 'Yes'))
    host_response_rate = st.slider('Host Response Rate', 0, 100, 0)
    host_acceptance_rate = st.slider('Host Acceptance Rate', 0, 100, 0)
with col2:
    availability = st.selectbox('Is the listing available?', ('Yes', 'No'))
    response = st.selectbox('Response time', (
    'within an hour', 'within a few hours', 'within a day', 'a few days or more'))


host_license =st.selectbox('Is your host licensed?', ('Yes', 'No'))
host_identity_verified = st.selectbox('Is your host verified?', ('Yes', 'No'))



st.markdown('---')
st.subheader("Guests' feedback")
col1, col2, col3 = st.columns(3)


with col1:
    location = st.slider('Location rating', 1.0, 5.0, 4.0, step=0.5)
    checkin = st.slider('Checkin rating', 1.0, 5.0, 3.0, step=0.5)

with col2:
    clean = st.slider('Cleanliness rating', 1.0, 5.0, 3.0, step=0.5)
    communication = st.slider('Communication rating', 1.0, 5.0, 4.0, step=0.5)
with col3:
    value_for_money_rating = st.slider('Value for money rating', 1.0, 5.0, 3.5, step=0.5)
    overall_rating = st.slider('Overall rating', 1.0, 5.0, 4.0, step=0.5)



def inputdatapreprocess_encoding(lat,lon):

    global data, key, value, input_test, haversine_distance, item, sia


    # Create a dictionary to store the data of all the inputs including the location of the marker and dates as well
    data = {
        'latitude': lat,
        'longitude': lon,
        'title': listing_title,
        'description': listing_description,
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': beds,
        'room_type': room_type,
        'instant_bookable': instant_bookable,
        'property_type': property_type,
        'minimum_nights': min_nights,
        'maximum_nights': max_nights,
        'availability_30': number_of_days_available_in_a_month,
        'availability_60': number_of_days_available_in_two_months,
        'availability_90': number_of_days_available_in_three_months,
        'availability_365': number_of_days_available_in_year,
        'host_since': host_registration_date,
        'host_is_superhost': super_host,
        'host_response_rate': host_response_rate,
        'host_acceptance_rate': host_acceptance_rate,
        'has_availability': availability,
        'host_response_time': response,
        'license': host_license,
        'host_identity_verified': host_identity_verified,
        'review_scores_location': location,
        'review_scores_checkin': checkin,
        'review_scores_cleanliness': clean,
        'review_scores_communication': communication,
        'review_scores_value_for_money': value_for_money_rating,
        'review_scores_rating': overall_rating
    }


    # st.write(data)


    input_test = pd.DataFrame([data])

    downtown_coords = (43.6532, -79.3832)
    lat, lon = downtown_coords

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # Calculate the distance to downtown for each entry in the dataset
    input_test['distance_to_downtown'] = input_test.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], lat, lon), axis=1
    )

    # Display the DataFrame
    st.markdown('---')

    st.subheader('Summary of Inputs')
    st.write(input_test)

    # print(input_test)
    input_test.dropna(inplace=True)

    today_date = datetime.today().strftime('%Y-%m-%d')
    # Converting date columns
    input_test['host_since'] = pd.to_datetime(input_test['host_since'])
    
    current_date = datetime.now()
    # Calculating values and storing in a new column
    input_test['host_since_days'] = (current_date - input_test['host_since']).dt.days
    
    # Dropping date columns
    input_test.drop(columns=['host_since'], inplace=True)

   
    # st.write(attraction_counts)
    # Converting boolean columns and picture url columns to 0s and 1s
    for boolean_column in ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        input_test[boolean_column] = input_test[boolean_column].map(lambda s: False if s == "f" else True)
    # Changing data in license column to licensed and unlicensed and converting to boolean
    input_test['license'] = input_test['license'].map(lambda s: False if s == "Unlicensed" else True)
    # print(input_test['host_response_time'])
    # Initialize OrdinalEncoder with the defined categories
    categories = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
    # Initialize OrdinalEncoder with the defined categories
    ordinal_encoder = OrdinalEncoder(categories=[categories])
    # Ordinal Encoding host_response_time since there is a clear order
    # Fit and transform the 'host_response_time' column
    input_test['host_response_time_encoded'] = ordinal_encoder.fit_transform(input_test[['host_response_time']])
    input_test.drop(columns=['host_response_time'], inplace=True)




    # creating an object of sentiment intensity analyzer
    # ensure_nltk_resources()

    # sia = SentimentIntensityAnalyzer()
    # creating new columns using polarity scores function
    input_test['title_scores'] = input_test['title'].apply(lambda title: sia.polarity_scores(str(title)))
    input_test['title_sentiment'] = input_test['title_scores'].apply(lambda score_dict: score_dict['compound'])
    input_test.drop(['title', 'title_scores'], axis=1, inplace=True)
    # creating new columns using polarity scores function
    input_test['description_scores'] = input_test['description'].apply(
        lambda description: sia.polarity_scores(str(description)))
    input_test['description_sentiment'] = input_test['description_scores'].apply(
        lambda score_dict: score_dict['compound'])
    input_test.drop(['description', 'description_scores'], axis=1, inplace=True)


    # # Convert the DataFrame to a CSV file in memory
    csv = input_test.to_csv(index=False).encode('utf-8')

    # Display the DataFrame
    # st.markdown('---')
    # st.subheader('Data preprocessing')
    # st.write(input_test)

    return input_test

input_test = inputdatapreprocess_encoding(lat,lon)


isPredictedPriceAvailable = False
#Create a select box for the model
with st.container():
    st.markdown('---')
    st.subheader('Model Selection')
    #Select the models from the user
    selected_model_name = st.selectbox('Select the model to predict the price', ('Ridge Regression', 'LightGBM', 'MLP'))

# Price Prediction button
_, col2, _ = st.columns(3)
with col2:
    #Add here multislect and add model names and predict the model using MSE,R2 score create a bar chart which one is the best and why
    run_preds = st.button('Predict the price')


    if run_preds:
        if selected_model_name == 'Ridge Regression':

            #Get unique values of property_type and room_type
            property_types = airbnb_df['property_type'].unique()
            room_types = airbnb_df['room_type'].unique()

            # st.write(st.session_state.dummy_vars_tp)
            # Create the dummy variables without dropping the original column
            dummy_vars = pd.get_dummies(property_types,prefix='property')
            dummy_vars = dummy_vars.reindex(columns=st.session_state.dummy_vars_tp.columns, fill_value=0)


            # Concatenate the original DataFrame with the dummy variables
            input_test = pd.concat([input_test, dummy_vars], axis=1)

            # st.write(st.session_state.dummy_vars_tr.columns)
            # Create the dummy variables without dropping the original column
            dummy_vars = pd.get_dummies(room_types, prefix='room_type')
            dummy_vars = dummy_vars.reindex(columns=st.session_state.dummy_vars_tr.columns, fill_value=0)

            # Concatenate the original DataFrame with the dummy variables
            input_test = pd.concat([input_test, dummy_vars], axis=1)
            input_test.drop(columns=['property_type', 'room_type'], inplace=True)

            # st.write(input_test.columns)

            # Drop all nan values
            input_test.dropna(inplace=True)

            # Feature Scaling
            scaler = StandardScaler()

            input_test = input_test[sorted(input_test.columns)]

            X_train_scaled = scaler.fit_transform(st.session_state.X_train)
            X_test_scaled = scaler.transform(input_test)

            # st.write(input_test)

            model = st.session_state.model['ridge'] #get the model from the session state
            predicted_price = model.predict(X_test_scaled)
            st.session_state.log_predicted_price = predicted_price

        elif selected_model_name == 'LightGBM':
            model = st.session_state.model['lightgbm']
              # # Encode categorical features for LightGBM
            categorical_features = ['property_type', 'room_type']
            for col in categorical_features:
                input_test[col] = input_test[col].astype('category')
            # Convert categorical features to their indices
            categorical_feature_indices = [input_test.columns.get_loc(col) for col in categorical_features]
            input_test = input_test[sorted(input_test.columns)]
            predicted_price = model.predict(input_test)
            st.session_state.log_predicted_price = predicted_price
        elif selected_model_name == 'MLP':
            # Get unique values of property_type and room_type
            property_types = airbnb_df['property_type'].unique()
            room_types = airbnb_df['room_type'].unique()

            # st.write(st.session_state.dummy_vars_tp)
            # Create the dummy variables without dropping the original column
            dummy_vars = pd.get_dummies(property_types, prefix='property')
            dummy_vars = dummy_vars.reindex(columns=st.session_state.dummy_vars_tp.columns, fill_value=0)

            # Concatenate the original DataFrame with the dummy variables
            input_test = pd.concat([input_test, dummy_vars], axis=1)

            # st.write(st.session_state.dummy_vars_tr.columns)
            # Create the dummy variables without dropping the original column
            dummy_vars = pd.get_dummies(room_types, prefix='room_type')
            dummy_vars = dummy_vars.reindex(columns=st.session_state.dummy_vars_tr.columns, fill_value=0)

            # Concatenate the original DataFrame with the dummy variables
            input_test = pd.concat([input_test, dummy_vars], axis=1)
            input_test.drop(columns=['property_type', 'room_type'], inplace=True)

            # st.write(input_test.columns)

            # Drop all nan values
            input_test.dropna(inplace=True)

            # Feature Scaling
            scaler = StandardScaler()

            input_test = input_test[sorted(input_test.columns)]

            X_train_scaled = scaler.fit_transform(st.session_state.X_train)
            X_test_scaled = scaler.transform(input_test)

            # st.write(input_test)

            model = st.session_state.model['mlp']  # get the model from the session state
            predicted_price = model.predict(X_test_scaled)
            st.session_state.log_predicted_price = predicted_price
        else:
            st.error('Model not found')


        # st.write(predicted_price)
        if predicted_price != 0:
            isPredictedPriceAvailable = True

        actual_predicted_price=int(np.exp(predicted_price))

        st.info(f"Predicted price is ${actual_predicted_price}")


# Creating a function to calculate SHAP values and display visualization based on provided parameters

def XAI_SHAP(model, data, graph, obs=0):
    """ Computes SHAP values and represents XAI graphs

    - Parameters:
        - model = Machine Learning model to interpret
        - data = Data used to make explanations
        - graph = Global or local interpretation
        - obs = Index of data instance to explain

    - Output:
        - XAI graphs and SHAP values
    """
    # Print JavaScript visualizations
    shap.initjs()

    # Create object to calculate SHAP values
    # if selected_model_name == 'Ridge Regression':
    #     # Create object to calculate SHAP values using LinearExplainer for Ridge model
    #     masker = shap.maskers.Independent(data)
    #     explainer = shap.LinearExplainer(model, masker)
    # elif selected_model_name == 'MLP':
    #     Create a masker using your training data (sample a subset if the data is large)
    #     masker = shap.maskers.Independent(data)
    #     # Initialize the SHAP KernelExplainer for your MLP model
    #     explainer = shap.KernelExplainer(model.predict, masker)
    # else:
    explainer = shap.Explainer(model)
    shap_values = explainer(data)


    if graph == 'global':
        # Global Interpretability (feature importance)
        shap.plots.bar(shap_values, max_display=20)

        # Global Interpretability (impact on target variable)
        shap.summary_plot(shap_values, data, max_display=20)
        
    else:
        plt.figure(figsize=(10, 5))
        # Local Interpretability (coefficients)
        # shap.plots.bar(shap_values[obs], max_display=20)
        shap.plots.waterfall(shap_values[obs], max_display=20)
        st.pyplot(plt.gcf())
        # Local Interpretability (force plots)
        # shap.plots.force(shap_values[obs])
    return shap_values

if 'log_predicted_price' in st.session_state:
    test = [input_test, st.session_state.log_predicted_price]

    # fig = plt.subplots(1, 1, figsize=(10, 20))
    # plt.figure(figsize=(10, 5))
if selected_model_name == 'LightGBM':
        st.header("Local Interpratibility")
        shap_values = XAI_SHAP(st.session_state.model['lightgbm'], test[0], 'local', 0)
else:
    pass
   

