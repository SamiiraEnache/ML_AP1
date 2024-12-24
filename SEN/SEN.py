import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def load_and_preprocess_data():
    """
Am incarcat datele din Excel si le-am procesat
    - Conversia coloanelor numerice
    - Conversia datelor din format text in datetime
    - Eliminarea randurilor fara valori (daca au existat)

    Returns:
        DataFrame procesat
    """
    file_path = r"C:\Users\Administrator\PycharmProjects\SEN\Grafic_SEN.xlsx"
    data = pd.read_excel(file_path, sheet_name="Grafic SEN")

    print("TEST(passed): Primele randuri ale dataset-ului:")
    print(data.head())
    print("\nInfo despre dataset:")
    print(data.info())

    numeric_columns = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]'
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data['Data'] = pd.to_datetime(data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

    data = data.dropna()

    print("\nPost preprocesare:")
    print(data.info())
    print("\nTEST:")
    print(data.head())

    return data


def metoda_1(data):
    """
    Metoda 1:
    - Predictia valorilor "Sold[MW]" folosind doar aceasta coloana

    Args:
        data: DataFrame procesat
    """
    sold_data = data[['Data', 'Sold[MW]']].copy()
    train_data = sold_data[sold_data['Data'].dt.month < 12].copy()
    test_data = sold_data[sold_data['Data'].dt.month == 12].copy()

    print(f"\nNr de randuri in setul de antrenare: {len(train_data)}")
    print(f"\nNr de randuri în setul de testare: {len(test_data)}")
    print("\nSet de antrenare:")
    print(train_data.head())
    print("\nSet de testare:")
    print(test_data.head())

    for df in [train_data, test_data]:
        df['hour'] = df['Data'].dt.hour
        df['day'] = df['Data'].dt.day
        df['month'] = df['Data'].dt.month
        df['dayofweek'] = df['Data'].dt.dayofweek

    train_data['Sold_Bucket'] = pd.cut(train_data['Sold[MW]'], bins=5, labels=False)

    X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
    X_test = test_data[['hour', 'day', 'month', 'dayofweek']]
    y_train = train_data['Sold_Bucket']

    # -------------------------------------------------
    # Modelul ID3
    # -------------------------------------------------
    id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    id3_model.fit(X_train, y_train)

    pred_buckets_id3 = id3_model.predict(X_test)

    bucket_edges = pd.cut(train_data['Sold[MW]'], bins=5).cat.categories
    bucket_means = bucket_edges.mid.values
    pred_values_id3 = [bucket_means[int(bucket)] for bucket in pred_buckets_id3]

    y_test_actual = test_data['Sold[MW]'].values

    # -------------------------------------------------
    # Modelul Bayesian
    # -------------------------------------------------
    for col in ['hour', 'day', 'month', 'dayofweek']:
        train_data[f'{col}_Bucket'] = pd.qcut(train_data[col], q=5, labels=False)
        test_data[f'{col}_Bucket'] = pd.cut(test_data[col], bins=5, labels=False)

    X_train_bayes = train_data[[f'{col}_Bucket' for col in ['hour', 'day', 'month', 'dayofweek']]]
    X_test_bayes = test_data[[f'{col}_Bucket' for col in ['hour', 'day', 'month', 'dayofweek']]]

    nb_model = GaussianNB()
    nb_model.fit(X_train_bayes, y_train)

    pred_buckets_bayes = nb_model.predict(X_test_bayes)
    pred_values_bayes = [bucket_means[int(bucket)] for bucket in pred_buckets_bayes]

    # -------------------------------------------------
    # Evaluarea performantei
    # -------------------------------------------------
    mse_id3 = mean_squared_error(y_test_actual, pred_values_id3)
    rmse_id3 = mse_id3 ** 0.5
    mae_id3 = mean_absolute_error(y_test_actual, pred_values_id3)

    mse_bayes = mean_squared_error(y_test_actual, pred_values_bayes)
    rmse_bayes = mse_bayes ** 0.5
    mae_bayes = mean_absolute_error(y_test_actual, pred_values_bayes)

    print(f"\nPerformanta modelului ID3:")
    print(f"RMSE: {rmse_id3:.2f}")
    print(f"MAE: {mae_id3:.2f}")

    print(f"\nPerformanta modelului Bayesian:")
    print(f"RMSE: {rmse_bayes:.2f}")
    print(f"MAE: {mae_bayes:.2f}")

    # -------------------------------------------------
    # Vizualizarea rezultatelor
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Data'], y_test_actual, label='Valori reale', color='blue')
    plt.plot(test_data['Data'], pred_values_id3, label='Predictii ID3', alpha=0.7, color='orange')
    plt.plot(test_data['Data'], pred_values_bayes, label='Predictii Bayesian', alpha=0.7, color='green')
    plt.xlabel('Data')
    plt.ylabel('Sold[MW]')
    plt.title('Compararea valorilor reale cu predictiile ID3 și Bayesian')
    plt.legend()
    plt.show()



def metoda_2(data):
    """
    Metoda 2:
    - Prezicerea fiecarei coloane (Consum[MW], Producție[MW], etc.)
    - Calcularea Sold[MW] ca diferenta între Producție[MW] și Consum[MW]


    Args:
        data: DataFrame procesat.
    """
    train_data = data[data['Data'].dt.month < 12].copy()
    test_data = data[data['Data'].dt.month == 12].copy()

    print(f"\nNr de randuri în setul de antrenare: {len(train_data)}")
    print(f"\nNr de randuri în setul de testare: {len(test_data)}")

    for df in [train_data, test_data]:
        df['hour'] = df['Data'].dt.hour
        df['day'] = df['Data'].dt.day
        df['month'] = df['Data'].dt.month
        df['dayofweek'] = df['Data'].dt.dayofweek

    target_columns = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
    ]

    predicted_columns_id3 = {}
    predicted_columns_bayes = {}

    # -------------------------------------------------
    # ID3
    # -------------------------------------------------
    for target in target_columns:
        print(f"\nAntrenarea modelului ID3 pentru coloana: {target}")

        X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
        y_train = train_data[target]
        X_test = test_data[['hour', 'day', 'month', 'dayofweek']]

        id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
        id3_model.fit(X_train, y_train)

        predicted_columns_id3[target] = id3_model.predict(X_test)

    # -------------------------------------------------
    # Model Bayesian
    # -------------------------------------------------
    for target in target_columns:
        print(f"\nAntrenarea modelului Bayesian pentru coloana: {target}")

        for col in ['hour', 'day', 'month', 'dayofweek']:
            train_data[f'{col}_Bucket'] = pd.cut(train_data[col], bins=10, labels=False)
            test_data[f'{col}_Bucket'] = pd.cut(test_data[col], bins=10, labels=False)

        X_train_bayes = train_data[[f'{col}_Bucket' for col in ['hour', 'day', 'month', 'dayofweek']]]
        y_train_bayes = train_data[target]
        X_test_bayes = test_data[[f'{col}_Bucket' for col in ['hour', 'day', 'month', 'dayofweek']]]

        bayes_model = GaussianNB()
        bayes_model.fit(X_train_bayes, y_train_bayes)

        predicted_columns_bayes[target] = bayes_model.predict(X_test_bayes)


    predicted_sold_id3 = (
            predicted_columns_id3['Productie[MW]'] - predicted_columns_id3['Consum[MW]']
    )
    predicted_sold_bayes = (
            predicted_columns_bayes['Productie[MW]'] - predicted_columns_bayes['Consum[MW]']
    )

    actual_sold = test_data['Sold[MW]'].values

    # -------------------------------------------------
    # Evaluarea perf pentru ID3
    # -------------------------------------------------
    mse_id3 = mean_squared_error(actual_sold, predicted_sold_id3)
    rmse_id3 = mse_id3 ** 0.5
    mae_id3 = mean_absolute_error(actual_sold, predicted_sold_id3)

    print(f"\nPerformanta modelului ID3 pentru Sold[MW]:")
    print(f"RMSE: {rmse_id3:.2f}")
    print(f"MAE: {mae_id3:.2f}")

    # -------------------------------------------------
    # Evaluarea perf pentru Bayesian
    # -------------------------------------------------
    mse_bayes = mean_squared_error(actual_sold, predicted_sold_bayes)
    rmse_bayes = mse_bayes ** 0.5
    mae_bayes = mean_absolute_error(actual_sold, predicted_sold_bayes)

    print(f"\nPerformanta modelului Bayesian pentru Sold[MW]:")
    print(f"RMSE: {rmse_bayes:.2f}")
    print(f"MAE: {mae_bayes:.2f}")


    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Data'], actual_sold, label='Valori reale')
    plt.plot(test_data['Data'], predicted_sold_id3, label='Predictii Sold[MW] ID3', alpha=0.7)
    plt.plot(test_data['Data'], predicted_sold_bayes, label='Predictii Sold[MW] Bayesian', alpha=0.7)
    plt.xlabel('Data')
    plt.ylabel('Sold[MW]')
    plt.title('Compararea valorilor reale cu predicțiile Sold[MW]')
    plt.legend()
    plt.show()


def metoda_3(data):
    """
    Metoda 3:
    - Agregarea productiei pe categorii: "Intermitenta" (solar și eolian) și "Constanta" (nuclear, hidro, etc.)
    - Predictia pentru fiecare categorie și pentru consum folosind ID3 și Bayesian
    - Calcularea Sold[MW] ca dif intre productia totala si consum

    Args:
        data: DataFrame procesat.
    """

    data['Producție_Intermitentă'] = data['Eolian[MW]'] + data['Foto[MW]']
    data['Producție_Constantă'] = (
        data['Nuclear[MW]'] + data['Hidrocarburi[MW]'] +
        data['Ape[MW]'] + data['Carbune[MW]'] + data['Biomasa[MW]']
    )

    train_data = data[data['Data'].dt.month < 12].copy()
    test_data = data[data['Data'].dt.month == 12].copy()

    print(f"\nNumăr de rânduri în setul de antrenare: {len(train_data)}")
    print(f"\nNumăr de rânduri în setul de testare: {len(test_data)}")

    for df in [train_data, test_data]:
        df['hour'] = df['Data'].dt.hour
        df['day'] = df['Data'].dt.day
        df['month'] = df['Data'].dt.month
        df['dayofweek'] = df['Data'].dt.dayofweek

    target_columns = ['Producție_Intermitentă', 'Producție_Constantă', 'Consum[MW]']
    predicted_columns_id3 = {}
    predicted_columns_bayes = {}

    # -------------------------------------------------
    # ID3
    # -------------------------------------------------
    for target in target_columns:
        print(f"\nAntrenarea modelului ID3 pentru coloana: {target}")

        X_train = train_data[['hour', 'day', 'month', 'dayofweek']]
        y_train = train_data[target]
        X_test = test_data[['hour', 'day', 'month', 'dayofweek']]

        id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
        id3_model.fit(X_train, y_train)

        predicted_columns_id3[target] = id3_model.predict(X_test)

    # -------------------------------------------------
    # Bayesian
    # -------------------------------------------------
    for target in target_columns:
        print(f"\nAntrenarea modelului Bayesian pentru coloana: {target}")

        for col in ['hour', 'day', 'month', 'dayofweek']:
            train_data[f'{col}_Bucket'] = pd.cut(train_data[col], bins=5, labels=False)
            test_data[f'{col}_Bucket'] = pd.cut(test_data[col], bins=5, labels=False)

        X_train_bayes = train_data[[f'{col}_Bucket' for col in ['hour', 'day', 'month', 'dayofweek']]]
        y_train_bayes = train_data[target]
        X_test_bayes = test_data[[f'{col}_Bucket' for col in ['hour', 'day', 'month', 'dayofweek']]]

        bayes_model = GaussianNB()
        bayes_model.fit(X_train_bayes, y_train_bayes)

        predicted_columns_bayes[target] = bayes_model.predict(X_test_bayes)


    predicted_production_id3 = (
        predicted_columns_id3['Producție_Intermitentă'] +
        predicted_columns_id3['Producție_Constantă']
    )
    predicted_sold_id3 = predicted_production_id3 - predicted_columns_id3['Consum[MW]']

    predicted_production_bayes = (
        predicted_columns_bayes['Producție_Intermitentă'] +
        predicted_columns_bayes['Producție_Constantă']
    )
    predicted_sold_bayes = predicted_production_bayes - predicted_columns_bayes['Consum[MW]']

    actual_sold = test_data['Sold[MW]'].values


    mse_id3 = mean_squared_error(actual_sold, predicted_sold_id3)
    rmse_id3 = mse_id3 ** 0.5
    mae_id3 = mean_absolute_error(actual_sold, predicted_sold_id3)

    print(f"\nPerformanța modelului ID3 pentru Sold[MW]:")
    print(f"RMSE: {rmse_id3:.2f}")
    print(f"MAE: {mae_id3:.2f}")


    mse_bayes = mean_squared_error(actual_sold, predicted_sold_bayes)
    rmse_bayes = mse_bayes ** 0.5
    mae_bayes = mean_absolute_error(actual_sold, predicted_sold_bayes)

    print(f"\nPerformanța modelului Bayesian pentru Sold[MW]:")
    print(f"RMSE: {rmse_bayes:.2f}")
    print(f"MAE: {mae_bayes:.2f}")


    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Data'], actual_sold, label='Valori reale')
    plt.plot(test_data['Data'], predicted_sold_id3, label='Predicții Sold[MW] ID3', alpha=0.7, color='orange')
    plt.plot(test_data['Data'], predicted_sold_bayes, label='Predicții Sold[MW] Bayesian', alpha=0.7, color='green')
    plt.xlabel('Data')
    plt.ylabel('Sold[MW]')
    plt.title('Compararea valorilor reale cu predictiile Sold[MW] - ID3 și Bayesian')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    selected_method = 3

    data = load_and_preprocess_data()

    if selected_method == 1:
        print("\nFolosind doar coloana 'Sold[MW]'")
        metoda_1(data)
    elif selected_method == 2:
        print("\nPrezicerea fiecărei coloane și calcLularea Sold[MW]")
        metoda_2(data)
    elif selected_method == 3:
        print("\nAgregarea producției pe categorii și calcularea Sold[MW]")
        metoda_3(data)

