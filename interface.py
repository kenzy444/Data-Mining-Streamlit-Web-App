import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from itertools import chain, combinations
import itertools
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from DT import DecisionTree
from Randomforest import RandomForest
from KNN import KNN
from Kmeans import KMeans



# Function to generate a histogram for the selected attribute
def show_histogram(data, selected_attribute):
    plt.figure(figsize=(16, 8))
    plt.hist(data[selected_attribute], bins='auto', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram for {selected_attribute}')
    plt.xlabel(selected_attribute)
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())
    plt.close()

def show_boxplot(data, selected_attribute):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[selected_attribute])
    plt.title(f'Box Plot for {selected_attribute}')
    plt.xlabel(selected_attribute)
    plt.ylabel('Value')
    st.pyplot(plt.gcf())
    plt.close()
# Function to generate a scatter plot for the selected attributes
def show_scatter_plot(data, x_attribute, y_attribute):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_attribute, y=y_attribute, data=data)
    plt.title(f'Scatter Plot for {x_attribute} and {y_attribute}')
    plt.xlabel(x_attribute)
    plt.ylabel(y_attribute)
    st.pyplot(plt.gcf())
    plt.close()

# Function to apply normalization and display a sample of the dataset
def apply_normalization(data, normalization_method):
    st.header("Normalization")

    if normalization_method == "Min-Max":
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    elif normalization_method == "Z-Score":
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    else:
        st.warning("Please select a normalization method.")

    st.write(f"Sample of Normalized Data ({normalization_method} Normalization)")
    st.write(normalized_data.head())

def preprocessing_page(data):
    st.title("Preprocessing")

    st.header("Histogram Visualization")

    # Display a select list with all attributes for histogram
    selected_histogram_attribute = st.selectbox("Select Attribute for Histogram", data.columns)

    # Show histogram for the selected attribute
    if selected_histogram_attribute:
        show_histogram(data, selected_histogram_attribute)

    st.header("Boxplots Visualization")

    # Display a select list with all attributes for boxplot
    selected_boxplot_attribute = st.selectbox("Select Attribute for Box Plot", data.columns)

    # Show box plot for the selected attribute
    if selected_boxplot_attribute:
        show_boxplot(data, selected_boxplot_attribute)

    st.header("Scatter Plot Visualization")

    # Display a select list with all attributes for scatter plot
    selected_x_attribute = st.selectbox("Select X Attribute for Scatter Plot", data.columns)
    selected_y_attribute = st.selectbox("Select Y Attribute for Scatter Plot", data.columns)

    # Show scatter plot for the selected attributes
    if selected_x_attribute and selected_y_attribute:
        show_scatter_plot(data, selected_x_attribute, selected_y_attribute)

    # Normalization section
    st.header("Normalization")
    d1 = pd.read_csv("data1.csv")
    normalization_method = st.radio("Select Normalization Method", ["Min-Max", "Z-Score"])

    if st.button("Apply Normalization"):
        apply_normalization(d1, normalization_method)


def visualize_distribution_by_zones(data2):
    st.subheader("The distribution of the total number of confirmed cases and positive tests by zones")

    # Convertir la colonne 'zcta' en type de données chaîne
    data2['zcta'] = data2['zcta'].astype(str)

    # Créer un DataFrame pour les deux attributs
    df_combined = pd.melt(data2, id_vars='zcta', value_vars=['case count', 'positive tests'])

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Afficher les deux attributs sur le même plot
    sns.barplot(x='zcta', y='value', hue='variable', data=df_combined, estimator=sum, ci=None, ax=ax)

    # Customize the plot
    ax.set_title('Distribution du nombre total de cas confirmés et de tests positifs par zone')
    ax.set_xlabel('Zone (zcta)')
    ax.set_ylabel('Nombre total')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Attribut')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

def visualize_weekly(data2):
    st.subheader("The evolution of COVID-19 tests, positive tests, and the number of cases over time weekly for a selected area")

    # Convert 'Start date' to datetime
    data2['Start date'] = pd.to_datetime(data2['Start date'], format='%Y-%m-%d')

    # Filter data for the zone 95129
    selected_zone = data2[data2['zcta'] == 95129]

    # Aggregate data by week
    selected_zone.loc[:, 'week'] = selected_zone['Start date'].dt.to_period('W')
    agg_data = selected_zone.groupby('week').agg({
        'test count': 'sum',
        'positive tests': 'sum',
        'case count': 'sum'
    }).reset_index()

    # Convert 'week' to string for plotting
    agg_data['week_str'] = agg_data['week'].astype(str)

    # Plot the line chart
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(agg_data['week_str'], agg_data['test count'], label='Test Count', marker='o')
    ax.plot(agg_data['week_str'], agg_data['positive tests'], label='Positive Tests', marker='o')
    ax.plot(agg_data['week_str'], agg_data['case count'], label='Case Count', marker='o')

    ax.set_title('Weekly Evolution of COVID-19 Tests, Positive Tests, and Cases for Zone 95129')
    ax.set_xlabel('Week')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    
    # Display the plot using Streamlit
    st.pyplot(fig)

def visualize_monthly(data2):
    st.subheader('The evolution of COVID-19 tests, positive tests, and the number of cases over time monthly for a selected area')

    # Convert 'Start date' to datetime
    data2['Start date'] = pd.to_datetime(data2['Start date'], format='%Y-%m-%d')

    # Filter data for the zone 95129
    selected_zone = data2[data2['zcta'] == 95129]

    # Aggregate data by month
    selected_zone.loc[:, 'year_month'] = selected_zone['Start date'].dt.to_period('M').astype(str)
    agg_data = selected_zone.groupby('year_month').agg({
        'test count': 'sum',
        'positive tests': 'sum',
        'case count': 'sum'
    }).reset_index()

    # Create a subplot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the line chart
    ax.plot(agg_data['year_month'], agg_data['test count'], label='Test Count', marker='o')
    ax.plot(agg_data['year_month'], agg_data['positive tests'], label='Positive Tests', marker='o')
    ax.plot(agg_data['year_month'], agg_data['case count'], label='Case Count', marker='o')

    ax.set_title('Monthly Evolution of COVID-19 Tests, Positive Tests, and Cases for Zone 95129')
    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

    # Display the plot using Streamlit
    st.pyplot(fig)


def visualize_anually(data2):
    st.subheader("The evolution of COVID-19 tests, positive tests, and the number of cases over time weekly for a selected area")

    # Convert 'zcta' to int and 'Start date' to datetime
    data2['zcta'] = data2['zcta'].astype(int)
    data2['test count'] = data2['test count'].astype(float)
    data2['positive tests'] = data2['positive tests'].astype(float)
    data2['case count'] = data2['case count'].astype(float)
    data2['Start date'] = pd.to_datetime(data2['Start date'], format='%Y-%m-%d')

    # Filter data for the zone 95129
    selected_zone = data2[data2['zcta'] == 95129]

    # Aggregate data by year
    selected_zone.loc[:, 'year'] = selected_zone['Start date'].dt.year  
    agg_data = selected_zone.groupby('year').agg({
        'test count': 'sum',
        'positive tests': 'sum',
        'case count': 'sum'
    }).reset_index()

    # Create a subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the line chart
    ax.plot(agg_data['year'], agg_data['test count'], label='Test Count', marker='o')
    ax.plot(agg_data['year'], agg_data['positive tests'], label='Positive Tests', marker='o')
    ax.plot(agg_data['year'], agg_data['case count'], label='Case Count', marker='o')
    ax.set_title('Annual Evolution of COVID-19 Tests, Positive Tests, and Cases for Zone 95129')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)

    # Set x-axis ticks explicitly as discrete years
    ax.set_xticks(agg_data['year'].unique())

    # Display the plot using Streamlit
    st.pyplot(fig)

def visualize_by_area(data2):
    st.subheader("The distribution of positive COVID-19 cases by area")

    # Convert 'zcta' to string
    data2['zcta'] = data2['zcta'].astype(str)

    # Create a DataFrame for positive cases by area
    df_positive_tests = data2.groupby(['zcta'])['positive tests'].sum().reset_index()

    # Plot the Stacked Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='zcta', y='positive tests', data=df_positive_tests, color='skyblue', ax=ax)
    ax.set_title('Distribution of positive COVID-19 cases by area')
    ax.set_xlabel('Zone (zcta)')
    ax.set_ylabel('Number of positive cases')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True)

    # Display the plot using Streamlit
    st.pyplot(fig)


def visualize_by_year(data2):
    st.subheader("The distribution of positive COVID-19 cases by year")

    # Convert date columns to datetime with the format month/day/year
    data2['Start date'] = pd.to_datetime(data2['Start date'], format='%Y-%m-%d')
    data2['end date'] = pd.to_datetime(data2['end date'], format='%Y-%m-%d')

    # Add a column to extract the year
    data2['Year'] = data2['Start date'].dt.year

    # Filter positive data
    positive_data = data2[data2['positive tests'].notna()]

    # Create a pivot DataFrame for positive cases by year
    pivot_data = positive_data.pivot_table(values='positive tests', index='Year', aggfunc='sum')

    # Create a Stacked Bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data.plot(kind='bar', stacked=True, ax=ax)

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of positive cases')
    ax.set_title('Distribution of positive COVID-19 cases by year')
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Display the plot using Streamlit
    st.pyplot(fig)

def visualize_ratio(data2):
    st.subheader("The ratio of the population to the number of tests performed")

    # Convert 'zcta' to string
    data2['zcta'] = data2['zcta'].astype(str)

    # Create a new column for the population-to-tests ratio
    data2['population_to_tests_ratio'] = data2['population'] / data2['test count']

    # Plot the scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(data2['population'], data2['test count'], c=data2['population_to_tests_ratio'], cmap='viridis', alpha=0.7)
    ax.set_title('Population-to-Tests Ratio by Zone')
    ax.set_xlabel('Population')
    ax.set_ylabel('Number of Tests')
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label='Population-to-Tests Ratio')

    # Display the plot using Streamlit
    st.pyplot(fig)

def visualize_by_5_areas(data2):
    st.subheader("The 5 areas most affected by the coronavirus")

    # Create a DataFrame for positive tests by area
    df_combined = pd.melt(data2, id_vars='zcta', value_vars=['positive tests'])

    # Group and sort the DataFrame by the total number of positive tests per zone
    df_combined = df_combined.groupby('zcta')['value'].sum().reset_index()
    df_combined = df_combined.sort_values(by='value', ascending=False)

    # Select the top five zones
    df_combined = df_combined.head(5)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='zcta', y='value', data=df_combined, ax=ax)
    ax.set_title('Distribution of the total number of positive tests by zone (Top 5 areas)')
    ax.set_xlabel('Zone (zcta)')
    ax.set_ylabel('Total Count')
    ax.grid(True)
    plt.xticks(rotation=45, ha='right')

    # Display the plot using Streamlit
    st.pyplot(fig)

def visualize_by_ratio2(data2):
    st.subheader("The ratio of confirmed cases, tests performed, and positive tests for each area")

    # Convert 'Start date' to datetime
    data2['Start date'] = pd.to_datetime(data2['Start date'], format='%Y-%m-%d')
    data2['end date'] = pd.to_datetime(data2['end date'], format='%Y-%m-%d')

    # Filter data for the chosen period (time_period = 61)
    selected_period_data = data2[data2['time_period'] == 61]

    # Create a pivot table for confirmed cases, tests performed, and positive tests by area
    pivot_data = selected_period_data.pivot_table(values=['case count', 'test count', 'positive tests'], index='zcta', aggfunc='sum')

    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data.plot(kind='bar', width=0.8, stacked=False, ax=ax)

    ax.set_xlabel('Zone (zcta)')
    ax.set_ylabel('Count')
    ax.set_title('Ratio of confirmed cases, tests performed, and positive tests (Period 31)')
    ax.legend(title='Attribute', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Display the plot using Streamlit
    st.pyplot(fig)



def visualisation_page(data2):
    st.title("Visualization")
    choice = ['The distribution of the total number of confirmed cases and positive tests by zones',
              'The evolution of COVID-19 tests, positive tests, and the number of cases over time weekly for area 95129',
              'The evolution of COVID-19 tests, positive tests, and the number of cases over time monthly for area 95129',
              'The evolution of COVID-19 tests, positive tests, and the number of cases over time anually for area 95129',
              'The distribution of positive COVID-19 cases by area',
              'The distribution of positive COVID-19 cases by year',
              'The ratio of the population to the number of tests performed',
              'The 5 areas most affected by the coronavirus',
              'The ratio of confirmed cases, tests performed, and positive tests for each area']

    # Display a select list with all attributes for discretization
    selected_choice = st.selectbox("Select what to visualize :", choice)

    if st.button("Visualize"):
        if selected_choice == 'The distribution of the total number of confirmed cases and positive tests by zones':
            visualize_distribution_by_zones(data2)
        if selected_choice == 'The evolution of COVID-19 tests, positive tests, and the number of cases over time weekly for area 95129':
            visualize_weekly(data2)
        if selected_choice == 'The evolution of COVID-19 tests, positive tests, and the number of cases over time monthly for area 95129':
            visualize_monthly(data2)
        if selected_choice == 'The evolution of COVID-19 tests, positive tests, and the number of cases over time anually for area 95129':
            visualize_anually(data2)
        if selected_choice == 'The distribution of positive COVID-19 cases by area':
            visualize_by_area(data2)
        if selected_choice == 'The distribution of positive COVID-19 cases by year':
            visualize_by_year(data2)
        if selected_choice == 'The ratio of the population to the number of tests performed':
            visualize_ratio(data2)
        if selected_choice == 'The 5 areas most affected by the coronavirus':
            visualize_by_5_areas(data2)
        if selected_choice == 'The ratio of confirmed cases, tests performed, and positive tests for each area':
            visualize_by_ratio2(data2)


# Function for discretization with equal width
def discretize_with_equal_width(data, column_to_discretize):
    # Define the number of classes (intervals)
    n = len(data)
    k = round(1 + (10 / 3) * np.log10(n))

    # Calculate the width of each interval
    value_min = data[column_to_discretize].min()
    value_max = data[column_to_discretize].max()
    interval_width = (value_max - value_min) / k

    # Create intervals
    intervals = [value_min + i * interval_width for i in range(k)]
    intervals.append(value_max)

    # Discretize the values of the column in-place
    data[column_to_discretize] = pd.cut(data[column_to_discretize], bins=intervals, labels=range(k))

    return data

# Function for discretization with equal frequency
def discretize_equal_frequency(data, column_to_discretize, num_bins):
    # Divide the data into classes with equal frequencies in-place
    data[column_to_discretize] = pd.qcut(data[column_to_discretize], q=num_bins, labels=range(num_bins))

    return data

def Apriori_algorithm(df, min_support):
    # Create a dictionary to store the support of each candidate itemset
    candidate_support = {}

    # Generate the 1-itemsets candidates (C1)
    c1 = {}
    for _, row in df.iterrows():
        items = [item.strip() for item in row['values'].split(",")]  # Remove leading and trailing spaces
        for item in items:
            if item not in c1:
                c1[item] = 1
            else:
                c1[item] += 1

    # Filter the 1-itemsets to keep only those with a support greater than or equal to the minimum support threshold
    lk1 = {item: count for item, count in c1.items() if count >= min_support}

    # Initialize the union dictionary (L) with items from L1
    L = lk1.copy()

    k = 2
    while True:
        # Generate the k-itemsets candidates (Ck) from Lk-1
        ck = {}
        for _, row in df.iterrows():
            items = [item.strip() for item in row['values'].split(",")]  # Remove leading and trailing spaces

            # Generate possible combinations of items of length k using itertools
            item_combinations = itertools.combinations(sorted(items), k)  # Sort items lexicographically

            # Update the support of the k-itemsets candidates
            for item_combination in item_combinations:
                item_combination_str = ",".join(item_combination)
                if item_combination_str not in ck:
                    ck[item_combination_str] = 1
                else:
                    ck[item_combination_str] += 1

        # Filter the k-itemsets to keep only those with a support greater than or equal to the minimum support threshold
        lk = {item: count for item, count in ck.items() if count >= min_support}

        if not lk:
            # If Lk is empty, stop the loop
            break

        # Update the union dictionary (L) with items from Lk
        L.update(lk)

        k += 1

    # Print the union of all Lk (L)    
    # print("L:----------------------------------------------------------------------------------------")
    for item, count in L.items():
        print(f"{item}: {count}")
    return L

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_association_rules(frequent_items, min_confidence):
    rules = []
    for itemset in frequent_items:
        if len(itemset) > 1:
            itemset_list = itemset.split(",")
            
            # Generate all possible combinations of antecedent and consequent
            itemset_powerset = list(powerset(itemset_list))
            for subset in itemset_powerset[1:-1]:  # Skip empty and full set
                antecedent = ",".join(sorted(subset))
                consequent = ",".join(sorted(set(itemset_list) - set(subset)))

                support_itemset = frequent_items[itemset]
                support_antecedent = frequent_items[antecedent]
                
                confidence = support_itemset / support_antecedent

                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
                    reverse_rule = (consequent, antecedent, confidence)
                    rules.append(reverse_rule)

    return rules


def apriori_page(data):
    st.title("Apriori")

    st.header("Discretization")
    columns_for_selection = ['Temperature', 'Humidity', 'Rainfall']
    # Display a select list with all attributes for discretization
    selected_discretize_attribute = st.selectbox("Select Attribute for Discretization", columns_for_selection)

    # Display radio buttons for choosing the discretization method
    discretization_method = st.radio("Select Discretization Method", ["Equal Width", "Equal Frequency"])

    # If Equal Width method is chosen, ask for the number of bins
    if discretization_method == "Equal Frequency":
        num_bins = st.slider("Number of Bins", min_value=2, max_value=20, value=5)
    else:
        num_bins = None

    # Show a sample of the dataset before discretization
    st.subheader("Sample of Data Before Discretization")
    st.write(data.head())

    # Apply discretization based on the chosen method
    if st.button("Apply Discretization"):
        if discretization_method == "Equal Width":
            data = discretize_with_equal_width(data, selected_discretize_attribute)
        elif discretization_method == "Equal Frequency":
            data = discretize_equal_frequency(data, selected_discretize_attribute, num_bins)

        # Show a sample of the dataset after discretization
        st.subheader("Sample of Data After Discretization")
        st.write(data.head())

    # Apriori section
    new_df = pd.DataFrame()
    new_df['id'] = data.index + 1  
    if discretization_method == "Equal Width":
            data = discretize_with_equal_width(data, selected_discretize_attribute)
    elif discretization_method == "Equal Frequency":
            data = discretize_equal_frequency(data, selected_discretize_attribute, num_bins)
    new_df['values'] = data.apply(lambda row: f"{row[selected_discretize_attribute]},{row['Soil']},{row['Crop']},{row['Fertilizer']}", axis=1)
    new_df = new_df.dropna(subset=['values'])

    st.header("Apply Apriori and Extract Association Rules")

    min_support = st.slider("Minimum Support", min_value=1, max_value=100, value=20)
    min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, step=0.1, value=0.5)

    st.subheader("Apriori Algorithm")
    frequent_items = Apriori_algorithm(new_df, min_support)

    # Button to generate association rules
    if st.button("Generate Association Rules"):
        st.subheader("Association Rules")
        association_rules = generate_association_rules(frequent_items, min_confidence)

        # Display the generated association rules
        st.write("Generated Association Rules:")
        for rule in association_rules:
            st.write(f"Rule :  {rule[0]} => {rule[1]}   |  Confidence: {rule[2]:.2f}")


# Function to evaluate classifier
def evaluate_classifier(classifier, X_train, X_test, Y_train, Y_test):
    start_time = time.time()

    # Fit the model on the training data
    classifier.fit(X_train, Y_train.flatten())

    # Make predictions on the test data
    Y_pred = classifier.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(Y_test, Y_pred)

    # Calculate class-specific metrics
    class_metrics = {}
    specificity_scores = {}
    unique_labels = np.unique(Y_test)
    for label in unique_labels:
     label_mask = Y_test.flatten() == label
     class_accuracy = accuracy_score(Y_test[label_mask], Y_pred[label_mask])
     class_precision = precision_score(Y_test[label_mask], Y_pred[label_mask], average='weighted')
     class_recall = recall_score(Y_test[label_mask], Y_pred[label_mask], average='weighted')
     class_f1 = f1_score(Y_test[label_mask], Y_pred[label_mask], average='weighted')
     # Calculate specificity
     tn, fp, fn, tp = confusion_matrix(Y_test == label, Y_pred == label).ravel()
     specificity = tn / (tn + fp)
     specificity_scores[f"Class {label}"] = specificity

     class_metrics[f"Class {label}"] = {
        'Accuracy': class_accuracy,
        'Precision': class_precision,
        'Recall': class_recall,
        'F1 Score': class_f1,
        'Specificity': specificity
     }
    global_specificity = np.mean(list(specificity_scores.values()))
    # Calculate execution time
    execution_time = time.time() - start_time

    result_str = f"Accuracy : {accuracy:.4f}\n"
    result_str += f"Precision : {precision:.4f}\n"
    result_str += f"Recall : {recall:.4f}\n"
    result_str += f"F1 Score : {f1:.4f}\n"
    result_str += f"Specificity : {global_specificity:.4f}\n"
    # Display the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Pass the figure to st.pyplot()
    st.pyplot(plt.gcf())
    result_str += "\nClass-specific Metrics:\n"
    for class_name, metrics in class_metrics.items():
        result_str += f"\n{class_name}:\n"
        for metric_name, value in metrics.items():
            result_str += f"{metric_name}: {value:.4f}\n"

    result_str += f"\nExecution Time: {execution_time:.4f} seconds"
    return result_str

def split_data_by_class(X, y, test_size=0.2, random_state=None):
    unique_classes = np.unique(y)
    X_train, X_test, Y_train, Y_test = [], [], [], []

    for class_label in unique_classes:
        # Filtrer les exemples pour la classe spécifique
        X_class = X[y.flatten() == class_label]
        y_class = y[y.flatten() == class_label]

        # Diviser les données pour cette classe
        X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(
            X_class, y_class, test_size=test_size, random_state=random_state
        )

        # Ajouter les données de cette classe à la liste globale
        X_train.append(X_train_class)
        X_test.append(X_test_class)
        Y_train.append(Y_train_class)
        Y_test.append(Y_test_class)

    # Concaténer les listes pour obtenir les données d'apprentissage et de test finales
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    Y_train = np.concatenate(Y_train)
    Y_test = np.concatenate(Y_test)

    return X_train, X_test, Y_train, Y_test

def decision_tree_page(data):
    st.title("Decision Tree")

    st.header("Execute Decision Tree")
    st.sidebar.subheader("Decision Tree Options")
    min_samples_split = st.sidebar.slider("Value of min_samples_split", min_value=1, max_value=50, value=3)
    max_depth = st.sidebar.slider("Value of max_depth", min_value=1, max_value=50, value=3)
    # Split the data
    X_train, X_test, Y_train, Y_test = split_data_by_class(data.iloc[:, :-1].values, data.iloc[:, -1].values.flatten(), test_size=0.2, random_state=41)
    classifier = DecisionTree(min_samples_split, max_depth)

    if st.button("Execute Decision Tree"):
        result_str = evaluate_classifier(classifier, X_train, X_test, Y_train, Y_test)

        # Display the results with HTML styling
        st.write(f"<div style='color:black; font-size:25px;'><b>Results:</b></div>", unsafe_allow_html=True)
        st.text(result_str)

        # Plot actual and predicted data
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot actual data with circles
        scatter_actual = ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test.flatten(), cmap="Paired", marker="o", label="Actual")

        # Plot predicted data with crosses
        classifier.fit(X_train, Y_train.flatten())
        prediction = classifier.predict(X_test)
        scatter_predicted = ax.scatter(X_test[:, 0], X_test[:, 1], c=prediction, cmap="Paired", marker="x", s=100, label="Predicted")

        ax.set_title('Comparison of Actual and Predicted Data with Decision Tree')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add legend
        ax.legend(handles=[scatter_actual, scatter_predicted], labels=['Actual', 'Predicted'], loc="upper right")

        plt.tight_layout()
        st.pyplot(fig)

    # Prediction Section
    st.header("Predict Class for Selected Instance")

    # Ensure that X_test is defined in this scope
    selected_instance_index = st.selectbox("Select an instance from X_test", range(len(X_test)))

    # Collect feature values for the selected instance
    selected_instance = X_test[selected_instance_index].reshape(1, -1)
    actual_class = Y_test[selected_instance_index]

    # Display the selected instance with attribute names
    st.subheader("Selected Instance:")
    st.write(f"Actual Class: {actual_class}")
    st.dataframe(pd.DataFrame([selected_instance.flatten()], columns=data.columns[:-1]))

    # Predict the class for the selected instance
    if st.button("Predict the Class"):
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(selected_instance)
        st.subheader("Prediction:")
        st.write(f"Predicted Class with DT : {prediction}")





def random_forest_page(data):
    st.title("Random Forest")

    st.sidebar.subheader("Random Forest Options")
    num_trees = st.sidebar.slider("Number of Trees", min_value=1, max_value=100, value=10)
    X_train, X_test, Y_train, Y_test = split_data_by_class(data.iloc[:, :-1].values, data.iloc[:, -1].values.flatten(), test_size=0.2, random_state=41)
    
    # Initialize RandomForest model
    random_forest = RandomForest(n_trees=num_trees)

    if st.button("Execute Random Forest"):
        st.subheader("Random Forest Results")

        # Split dataset into training and testing data
        random_forest.fit(X_train, Y_train.flatten())
        prediction = random_forest.predict(X_test)
        # Evaluate the model
        results = evaluate_classifier(random_forest, X_train, X_test, Y_train, Y_test)
        # Display the results with HTML styling
        st.write(f"<div style='color:black; font-size:25px;'><b>Results:</b></div>", unsafe_allow_html=True)
        st.text(results)

        # Plot actual and predicted data
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot actual data with circles
        scatter_actual = ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test.flatten(), cmap="Paired", marker="o", label="Actual")

        # Plot predicted data with crosses
        scatter_predicted = ax.scatter(X_test[:, 0], X_test[:, 1], c=prediction, cmap="Paired", marker="x", s=100, label="Predicted")

        ax.set_title('Comparison of Actual and Predicted Data with Random Forest ')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add legend
        ax.legend(handles=[scatter_actual, scatter_predicted], labels=['Actual', 'Predicted'], loc="upper right")

        plt.tight_layout()
        st.pyplot(fig)

    # Prediction Section
    st.header("Predict Class for Selected Instance")

    # Ensure that X_test is defined in this scope
    selected_instance_index = st.selectbox("Select an instance from X_test", range(len(X_test)))

    # Collect feature values for the selected instance
    selected_instance = X_test[selected_instance_index].reshape(1, -1)
    actual_class = Y_test[selected_instance_index]

    # Display the selected instance
    st.subheader("Selected Instance:")
    st.write(f"Actual Class: {actual_class}")
    st.dataframe(pd.DataFrame([selected_instance.flatten()], columns=data.columns[:-1]))

    # Predict the class for the selected instance
    if st.button("Predict the Class"):
        random_forest.fit(X_train, Y_train)
        prediction = random_forest.predict(selected_instance)
        st.subheader("Prediction:")
        st.write(f"Predicted Class with RF : {prediction}")


def k_nearest_neighbors_page(data):
    st.title("K-Nearest Neighbors")

    st.sidebar.subheader("KNN Options")
    k_value = st.sidebar.slider("Value of k", min_value=1, max_value=20, value=3)
    
    # Initialize KNN model
    knn = KNN(k=k_value)
    X_train, X_test, Y_train, Y_test = split_data_by_class(data.iloc[:, :-1].values, data.iloc[:, -1].values.flatten(), test_size=0.2, random_state=41)
    
    if st.button("Execute K-Nearest Neighbors"):
        st.subheader("K-Nearest Neighbors Results")

        # Split dataset into training and testing data
        # X_train, X_test, Y_train, Y_test = split_data_by_class(data.iloc[:, :-1].values, data.iloc[:, -1].values.flatten(), test_size=0.2, random_state=41)
        # Train the model
        knn.fit(X_train, Y_train)
        prediction = knn.predict(X_test)
        # Evaluate the model
        results = evaluate_classifier(knn, X_train, X_test, Y_train, Y_test)
        # Display the results with HTML styling
        st.write(f"<div style='color:black; font-size:25px;'><b>Results:</b></div>", unsafe_allow_html=True)
        st.text(results)
        # Plot actual and predicted data
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot actual data with circles
        scatter_actual = ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test.flatten(), cmap="Paired", marker="o", label="Actual")

        # Plot predicted data with crosses
        scatter_predicted = ax.scatter(X_test[:, 0], X_test[:, 1], c=prediction, cmap="Paired", marker="x", s=100, label="Predicted")

        ax.set_title('Comparison of Actual and Predicted Data with K-Nearest Neighbors')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Add legend
        ax.legend(handles=[scatter_actual, scatter_predicted], labels=['Actual', 'Predicted'], loc="upper right")

        plt.tight_layout()
        st.pyplot(fig)

    # Prediction Section
    st.header("Predict Class for Selected Instance")

    # Ensure that X_test is defined in this scope
    selected_instance_index = st.selectbox("Select an instance from X_test", range(len(X_test)))

    # Collect feature values for the selected instance
    selected_instance = X_test[selected_instance_index].reshape(1, -1)
    actual_class = Y_test[selected_instance_index]

    # Display the selected instance
    st.subheader("Selected Instance:")
    st.write(f"Actual Class: {actual_class}")
    st.dataframe(pd.DataFrame([selected_instance.flatten()], columns=data.columns[:-1]))

    # Predict the class for the selected instance
    if st.button("Predict the Class"):
        knn.fit(X_train, Y_train)
        prediction = knn.predict(selected_instance)
        st.subheader("Prediction:")
        st.write(f"Predicted Class with KNN : {prediction}")



def k_means_page():
    st.title("K-Means")
    
    # Load your CSV dataset
    file_path = "preprocessed_dataset1.csv"
    df = pd.read_csv(file_path)
    df = df.iloc[:, :-1]
    # Extract features from the DataFrame
    X_train = df.values
    
    st.sidebar.subheader("KMeans Options")
    centers = st.sidebar.slider("Value of centers", min_value=1, max_value=20, value=3)
    
    if st.button("Execute K-Means"):
        # Fit centroids to dataset
        kmeans = KMeans(n_clusters=centers)
        kmeans.fit(X_train)

        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        st.write(f"Number of centers : {centers}")
        # Silhouette Score
        silhouette_avg = silhouette_score(X_train_pca, kmeans.evaluate(X_train)[1])
        st.write(f"Silhouette Score : {silhouette_avg:.4f}")
        

        # View results with reduced dimensions
        classification = kmeans.evaluate(X_train)[1]
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_train_pca[:, 0],
                        y=X_train_pca[:, 1],
                        hue=classification,
                        palette="deep",
                        legend=None,
                        ax=ax
                        )
        plt.title('K-Means Clustering')
        plt.xlabel('')
        plt.ylabel('')
        st.pyplot(fig)


def custom_distance_matrix(data, metric='euclidean'):
    n = len(data)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'euclidean':
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
            # Add other distance metrics as needed

    # Fill in the lower triangular part of the matrix
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix

def custom_pairwise_distances(data, metric='euclidean'):
    n = len(data)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'euclidean':
                distances[i, j] = np.linalg.norm(data[i] - data[j])
            # Add other distance metrics as needed

    # Fill in the lower triangular part of the matrix
    distances = distances + distances.T

    return distances


def custom_dbscan(normalized_distance, e, k):
    DistanceMatrix = custom_distance_matrix(normalized_distance, metric='euclidean')
    core_point_array = np.zeros(len(normalized_distance))
    cluster_array = np.zeros(len(normalized_distance))
    PointNeighbors = []
    w = 0

    for i in range(len(DistanceMatrix)):
        PointNeighbors = np.where(DistanceMatrix[i] <= e)[0]
        if len(PointNeighbors) >= k:
            core_point_array[i] = 1
            if cluster_array[i] == 0:
                cluster_array[i] = w
                w += 1
            for x in range(len(PointNeighbors)):
                if cluster_array[PointNeighbors[x]] == 0:
                    cluster_array[PointNeighbors[x]] = cluster_array[i]

    for x in range(len(cluster_array)):
        cluster_array[x] = cluster_array[x] - 1

    return cluster_array

def prepare_data(input_data):
    poly = PolynomialFeatures(4)
    input_data = poly.fit_transform(input_data)
    input_data = QuantileTransformer(n_quantiles=40, random_state=0).fit_transform(input_data)

    scaler = MinMaxScaler()
    scaler.fit(input_data)
    normalized_input_data = scaler.transform(input_data)

    distan = custom_pairwise_distances(normalized_input_data, metric='euclidean')

    scaler.fit(distan)
    normalized_distance = scaler.transform(distan)

    # sscaler = StandardScaler()
    # sscaler.fit(normalized_distance)
    # normalized_distance = sscaler.transform(normalized_distance)

    pca = PCA(n_components=4)
    normalized_distance = pca.fit_transform(normalized_distance)

    scaler.fit(normalized_distance)
    normalized_distance = scaler.transform(normalized_distance)

    return normalized_distance

def dbscan_page():
    st.title("DBSCAN")

    st.sidebar.subheader("DBSCAN Options")
    # Sliders for DBSCAN parameters
    e = st.sidebar.slider("Epsilon (e)", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
    k = st.sidebar.slider("Min Points (k)", min_value=1, max_value=50, value=20, step=1)

    # Load CSV dataset
    file_path = "data1.csv"
    dataset = pd.read_csv(file_path)

    # Extract features and target
    input_data = dataset.drop(columns=['Fertility'])

    # Data Manipulations before introducing to the algorithm
    normalized_distance = prepare_data(input_data)

    # Execute DBSCAN on button click
    if st.button("Execute DBSCAN"):
        # Execute DBSCAN
        cluster_array = custom_dbscan(normalized_distance, e, k)
        st.write(f"Epsilon (e) : {e}")
        st.write(f"Min Points (k) : {k}")
        # Calculate silhouette score
        num_clusters = len(np.unique(cluster_array))

        if num_clusters > 1:
          silhouette_avg = silhouette_score(normalized_distance, cluster_array)
          st.write(f"Silhouette Score: {silhouette_avg:.4f}")
        else :
          st.write("There is only one cluster, silhouette score cannot be claculated !")
        # Plot results
        fig, ax = plt.subplots()
        ax.scatter(normalized_distance[:, 0], normalized_distance[:, 1], c=cluster_array, cmap='Paired')
        ax.set_title("Custom DBSCAN Predicted Cluster Outputs")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Display the plot using Streamlit
        st.pyplot(fig)




def welcome_page():
    st.title("DATASET MINER")
    st.image("dm3.jpeg", use_column_width=True)

    st.markdown(
        """
        Welcome to DataMiner! 🚀

        DataMiner is a powerful tool for exploring and analyzing datasets.
        It provides various functionalities to perform clustering, classification, and more.

        Let's get started by exploring dataset!

        """
    )

    if st.button("Start Analysis", key="start_button"):
        st.session_state.page = 1

def main():
    data = pd.read_csv('data1.csv')
    data2 =  pd.read_csv('data2.csv')
    data3 = pd.read_excel('Dataset3.xlsx')
    preprocced_data = pd.read_csv("preprocessed_dataset1.csv")
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 0

    if st.session_state.page == 0:
        welcome_page()
    elif st.session_state.page == 1:
        st.sidebar.title("Menu")

        menu_options = ["Static data","Temporal data", "Apriori", "Classification", "Clustering"]
        selected_option = st.sidebar.radio("Select an option", menu_options)

        if selected_option == "Static data":
            preprocessing_page(data)
        elif selected_option == "Temporal data":
            visualisation_page(data2)
        elif selected_option == "Apriori":
            apriori_page(data3)
        elif selected_option == "Classification":
            st.sidebar.subheader("Classification Methods")
            classification_options = ["Decision Tree", "Random Forest", "K-Nearest Neighbors"]
            selected_classification_option = st.sidebar.radio("Select a method", classification_options)

            if selected_classification_option == "Decision Tree":
                decision_tree_page(preprocced_data)
            elif selected_classification_option == "Random Forest":
                random_forest_page(preprocced_data)
            elif selected_classification_option == "K-Nearest Neighbors":
                k_nearest_neighbors_page(preprocced_data)
        elif selected_option == "Clustering":
            st.sidebar.subheader("Clustering Methods")
            clustering_options = ["K-Means", "DBSCAN"]
            selected_clustering_option = st.sidebar.radio("Select a method", clustering_options)

            if selected_clustering_option == "K-Means":
                k_means_page()
            elif selected_clustering_option == "DBSCAN":
                dbscan_page()


if __name__ == "__main__":
    main()