from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import resample
from treeinterpreter import treeinterpreter as ti


def display_imbalanced_data(data: pd.DataFrame, column: str) -> None:
    """
    Visualize the distribution of imbalanced data in the dataset.

    Args:
      data (pd.DataFrame): The dataset containing imbalanced column.
      column (str): column to display imbalanced data.
    """
    plt.figure(figsize=(10, 6))

    # Sort the unique values of the column in ascending order and remove nan values
    values = [
        value
        for value in sorted(list(data[column].astype(str).unique()))
        if value != "nan"
    ]

    # Create an array of random colors using seaborn
    colors = sns.color_palette("husl", len(values))

    # Normalize value counts and plot the distribution of RainTomorrow
    data[column].value_counts(normalize=True).plot(
        kind="bar",
        color=colors,
        alpha=0.9,
        rot=0,
        tick_label=values,
    )

    plt.title(f"""{column} Indicator in the Imbalanced Dataset""")
    plt.legend(values)
    plt.show()


def balance_dataset(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Balance the dataset by oversampling the minority class.

    Args:
      data (pd.DataFrame): The dataset containing RainTomorrow column.
      column (str): column to group the dataset by.

    Returns:
      pd.DataFrame: The balanced dataset.
    """
    # Separate the dataset based on RainTomorrow values
    class_groups = data.groupby(column)

    # Find the unique values of the column
    values = data[column].unique()

    # Find the majority class and minority class
    majority_class, minority_class = (
        (
            class_groups.get_group(values[0]),
            class_groups.get_group(values[1]),
        )
        if len(class_groups.get_group(values[0]))
        > len(class_groups.get_group(values[1]))
        else (
            class_groups.get_group(values[1]),
            class_groups.get_group(values[0]),
        )
    )

    # Oversample the minority class
    minority_class_oversampled = resample(
        minority_class,
        replace=True,
        n_samples=len(majority_class),
        random_state=42,
    )

    # Concatenate the majority class and the oversampled minority class
    balanced_data = pd.concat([majority_class, minority_class_oversampled])

    return balanced_data


def find_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Find the missing data in the dataset.

    Args:
      data (pd.DataFrame): The dataset to find missing data in.

    Returns:
      pd.DataFrame: A dataframe with total and percent missing data.
    """
    # Calculate the total number of missing values per column
    missing_count = data.isnull().sum()

    # Calculate the percentage of missing values per column
    missing_percentage = missing_count / len(data)

    # Combine the missing count and percentage into a dataframe
    missing_data = pd.DataFrame({"Total": missing_count, "Percent": missing_percentage})

    # Sort the dataframe by the total number of missing values
    missing_data_sorted = missing_data.sort_values(by="Total", ascending=False)

    return missing_data_sorted


def fill_categorical_missing_data(data: pd.DataFrame) -> None:
    """
    Fill the missing data in categorical columns using mode.

    Args:
      data (pd.DataFrame): The dataset with categorical missing data.
    """
    # Get categorical columns
    categorical_columns = data.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        # Fill missing data with the mode of the column
        data[column].fillna(data[column].mode()[0], inplace=True)


def convert_categorical_to_numerical(data: pd.DataFrame) -> None:
    """
    Convert categorical columns to numerical using Label Encoding.

    Args:
      data (pd.DataFrame): The dataset with categorical columns.
    """
    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        # Encode categorical data using label encoding
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])


def apply_iterative_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply iterative imputation to fill missing data in the dataset.

    Args:
      data (pd.DataFrame): The dataset with missing data.

    Returns:
      pd.DataFrame: The imputed dataset.
    """
    # Create a deep copy of the input data
    imputed_data = data.copy(deep=True)
    # Initialize iterative imputer
    imputer = IterativeImputer()
    # Impute missing data using iterative imputation
    imputed_data.iloc[:, :] = imputer.fit_transform(data)

    return imputed_data


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers in the dataset using the IQR method.

    Args:
      data (pd.DataFrame): The dataset with potential outliers.

    Returns:
      pd.DataFrame: The dataset without outliers.
    """
    # Get 25% and 75% quantiles
    lower_quantile = data.quantile(0.25)
    upper_quantile = data.quantile(0.75)
    interquartile_range = upper_quantile - lower_quantile
    # Keep data within the IQR range
    no_outliers = data[
        (
            (data >= (lower_quantile - 1.5 * interquartile_range))
            & (data <= (upper_quantile + 1.5 * interquartile_range))
        ).any(axis=1)
    ]

    return no_outliers


def display_correlation_heatmap(data: pd.DataFrame) -> None:
    """
    Display a heatmap of the correlation matrix of the dataset.

    Args:
      data (pd.DataFrame): The dataset to visualize correlations.
    """
    correlation_matrix = data.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=np.bool))
    plt.subplots(figsize=(20, 20))
    color_map = sns.diverging_palette(100, 15, as_cmap=True)
    # Plot the correlation heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap=color_map,
        vmax=None,
        center=0,
        square=True,
        annot=True,
        linewidths=0.25,
        cbar_kws={"shrink": 0.9},
    )
    plt.show()


def select_important_features(data: pd.DataFrame, target: str) -> List[str]:
    """
    Select the most important features in the dataset using SelectKBest.

    Args:
      data (pd.DataFrame): The dataset to analyze.
      target (str): The name of the target variable in the dataset.

    Returns:
      pd.Index: The column names of the selected important features.
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = pd.DataFrame(
        scaler.transform(data), index=data.index, columns=data.columns
    )
    # Get the feature matrix and target variable
    X = scaled_data.loc[:, scaled_data.columns != target]
    y = scaled_data[[target]]
    selector = SelectKBest(chi2, k=8)
    selector.fit(X, y)
    # Get the column names of the selected features
    return X.columns[selector.get_support(indices=True)]


def split_and_scale_dataset(
    features: pd.DataFrame, target: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets and scale the features.

    Args:
      features (pd.DataFrame): The feature matrix.
      target (pd.Series): The target variable.

    Returns:
      tuple: The split and scaled training and testing sets (X_train, X_test, y_train, y_test).
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def plot_receiver_operating_characteristic(
    false_positive_rate: np.ndarray, true_positive_rate: np.ndarray
) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
      false_positive_rate (array): False positive rates.
      true_positive_rate (array): True positive rates.
    """
    plt.plot(false_positive_rate, true_positive_rate, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


def evaluate_trained_model(
    model: RandomForestClassifier,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the trained model's performance.

    Args:
      model: The trained machine learning model.
      X_test (array): The testing set feature matrix.
      y_test (array): The testing set target variable.
      y_pred (array): The predicted target variable for the testing set.

    Returns:
      tuple: Accuracy, ROC AUC, Cohen's Kappa, False positive rates, and True positive rates.
    """
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(cohen_kappa))
    print(classification_report(y_test, y_pred, digits=5))

    # Calculate probabilities for ROC curve
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probs)

    plot_receiver_operating_characteristic(false_positive_rate, true_positive_rate)
    cm = confusion_matrix(y_test, y_pred, normalize="all")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    return accuracy, roc_auc, cohen_kappa, false_positive_rate, true_positive_rate


def perform_grid_search_cv(
    model, X, y, param_grid, num_folds=5, scoring_metric="accuracy"
):
    """
    Perform grid search cross-validation on the given model and dataset.

    Args:
      model: The model to evaluate.
      X: The feature set.
      y: The target set.
      param_grid: The dictionary of hyperparameters to search.
      num_folds: The number of folds for cross-validation.
      scoring_metric: The scoring metric to use for evaluation.

    Returns:
      gs: The fitted GridSearchCV object.
    """
    # Initialize GridSearchCV object with the given parameters
    gs = GridSearchCV(
        model, param_grid, cv=num_folds, scoring=scoring_metric, n_jobs=-1
    )

    # Perform grid search cross-validation
    gs.fit(X, y)

    return gs


def plot_rainfall_distribution(data: pd.DataFrame) -> None:
    """
    Plot a histogram of the Rainfall column.

    Args:
        data (pd.DataFrame): The dataset containing the Rainfall column.

    Returns:
        None
    """
    plt.hist(data["Rainfall"], bins=50, color="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rainfall")
    plt.show()


def plot_temperature_vs_rain(data: pd.DataFrame) -> None:
    """
    Plot a scatter plot of MinTemp and MaxTemp vs. RainTomorrow.

    Args:
        data (pd.DataFrame): The dataset containing the MinTemp, MaxTemp, and RainTomorrow columns.

    Returns:
        None
    """
    sns.scatterplot(x="MinTemp", y="MaxTemp", hue="RainTomorrow", data=data)
    plt.xlabel("Minimum Temperature (°C)")
    plt.ylabel("Maximum Temperature (°C)")
    plt.title("Temperature vs. RainTomorrow")
    plt.show()


def plot_windgustspeed_vs_rain(data: pd.DataFrame) -> None:
    """
    Plot a box plot of WindGustSpeed vs. RainTomorrow.

    Args:
        data (pd.DataFrame): The dataset containing the WindGustSpeed and RainTomorrow columns.

    Returns:
        None
    """
    sns.boxplot(x="RainTomorrow", y="WindGustSpeed", data=data)
    plt.xlabel("Rain Tomorrow")
    plt.ylabel("Wind Gust Speed (km/h)")
    plt.title("Wind Gust Speed vs. RainTomorrow")
    plt.show()


def feature_contributions(
    model: RandomForestClassifier, instance: pd.DataFrame, feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the contributions of each feature for a specific instance
    using the treeinterpreter package with a trained RandomForestClassifier model.

    Args:
        model (RandomForestClassifier): A trained RandomForestClassifier model.
        instance (pd.DataFrame): A single instance from the dataset (1 row of features) as a DataFrame.
        feature_names (List[str]): A list of feature names corresponding to the instance.

    Returns:
        pd.DataFrame: A pandas dataframe with the feature names, bias and prediction as columns and a single row with the feature contribution value, bias and predicted value for that instance.
    """

    # Use treeinterpreter to predict and obtain bias and feature contributions
    prediction, bias, contributions = ti.predict(model, instance)

    # Print prediction, bias, and feature contributions
    print("Prediction", prediction)
    print("Bias (trainset prior)", bias)
    print("Feature contributions:")

    # Create a DataFrame with feature contributions, bias, and prediction
    contributions_0 = [c[0] for c in contributions[0]]
    contributions_0.extend([bias[0][0], prediction[0][0]])

    contributions_1 = [c[1] for c in contributions[0]]
    contributions_1.extend([bias[0][1], prediction[0][1]])

    # Create a DataFrame with columns for features, bias, and prediction
    result_df = pd.DataFrame(
        [contributions_0, contributions_1],
        columns=feature_names.tolist() + ["Bias", "Prediction"],
        index=["No Rain", "Rain"],
    )

    return result_df


def process_train_evaluate_model(dataset: str, target: str) -> None:
    """
    Load data, split, scale and filter dataset, and train a RandomForestClassifier using
    GridSearchCV for hyperparameter tuning and cross-validation.
    """
    data = pd.read_csv(dataset)
    display_imbalanced_data(data, target)

    balanced_data = balance_dataset(data, target)

    # Display heatmap of missing values
    sns.heatmap(balanced_data.isnull(), cbar=False)

    plot_rainfall_distribution(balanced_data)
    plot_temperature_vs_rain(balanced_data)
    plot_windgustspeed_vs_rain(balanced_data)

    # Find missing data
    print(find_missing_data(balanced_data))

    # Fill and convert categorical missing data
    fill_categorical_missing_data(balanced_data)
    convert_categorical_to_numerical(balanced_data)

    # Impute missing data and remove outliers
    imputed_data = apply_iterative_imputation(balanced_data)
    cleaned_data = remove_outliers(imputed_data)

    # Display correlation heatmap
    display_correlation_heatmap(cleaned_data)

    # Select important features
    important_features = select_important_features(cleaned_data, target)

    # Split dataset into features and target
    features = cleaned_data[important_features]
    target = cleaned_data[target]

    # Split and scale dataset
    X_train, X_test, y_train, y_test = split_and_scale_dataset(features, target)

    # RandomForestClassifier parameters
    # params_random_forest = {
    #     "max_depth": [None, 10, 16, 20],
    #     "min_samples_leaf": [1, 2, 4],
    #     "min_samples_split": [2, 5, 10],
    #     "n_estimators": [50, 100, 150],
    #     "random_state": [42],
    # }

    # Using best performing hyperparameters since it takes a long time to run
    params_random_forest = {
        "max_depth": [16],
        "min_samples_leaf": [2],
        "min_samples_split": [2],
        "n_estimators": [100],
        "random_state": [42],
    }

    # Create and train RandomForestClassifier
    model_random_forest = RandomForestClassifier()

    # Perform grid search cross-validation
    grid_search_cv = perform_grid_search_cv(
        model_random_forest, X_train, y_train, params_random_forest
    )

    # Get the best model and its parameters
    best_model = grid_search_cv.best_estimator_
    best_params = grid_search_cv.best_params_
    print("Best parameters: ", best_params)

    y_pred = best_model.predict(X_test)

    # Evaluate trained model
    evaluate_trained_model(best_model, X_test, y_test, y_pred)

    instance = X_test[0:1]
    contributions_df = feature_contributions(best_model, instance, features.columns)
    print(contributions_df.head())


process_train_evaluate_model("weatherAUS.csv", "RainTomorrow")
