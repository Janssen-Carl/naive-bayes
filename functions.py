# functions.py for Job Placement Dataset Analysis using Naive Bayes Algorithm

# Loading the dataset

def load_data(filename):
    data = []
    
    with open(filename, "r") as file:
        lines = file.readlines()
        
        headers = lines[0].strip().split(",")
        
        for line in lines[1:]:
            values = line.strip().split(",")
            row = {}
            
            for i in range(len(headers)):
                row[headers[i].strip().lower().replace(" ", "_")] = values[i].strip()
            
            data.append(row)
    
    return data

# Data Cleaning 

#Convert numeric columns to float
def convert_numeric(data):
    numeric_cols = [
        "grade_point_average",
        "aptitude_score",
        "communication_score",
        "internship_quality_score"
    ]
    
    for row in data:
        for col in numeric_cols:
            row[col] = float(row[col])
    
    return data

# function 2.2 Clean categorical columns by stripping whitespace
def clean_categorical(data):
    categorical_cols = [
        "college_category",
        "country_of_residence",
        "university_ranking",
        "specialization",
        "industry",
        "placement_status"
    ]
    
    for row in data:
        for col in categorical_cols:
            row[col] = row[col].strip()
    
    return data

# Binning to numeric columns
def bin_gpa(x):
    if x < 5.5:
        return "poor"
    elif x < 7:
        return "average"
    elif x < 9:
        return "good"
    else:
        return "excellent"

# Binning aptitude score into 5-point intervals
def bin_aptitude(x):
    if x < 50:
        return "poor"
    elif x < 65:
        return "average"
    elif x < 80:
        return "good"
    else:
        return "excellent" 

# Binning communication score into 5-point intervals
def bin_communication(x):
    if x < 50:
        return "poor"
    elif x < 65:
        return "average"
    elif x < 80:
        return "good"
    else:
        return "excellent"

# Binning internship quality score into 5-point intervals
def bin_internship_quality(x):
    if x < 5.5:
        return "poor"
    elif x < 7:
        return "average"
    elif x < 9:
        return "good"
    else:
        return "excellent"

# Apply binning to the dataset
def apply_binning(data):
    for row in data:
        row["gpa_bin"] = bin_gpa(row["grade_point_average"])
        row["aptitude_bin"] = bin_aptitude(row["aptitude_score"])
        row["communication_bin"] = bin_communication(row["communication_score"])
        row["internship_quality_bin"] = bin_internship_quality(row["internship_quality_score"])
    
    return data

# Remove original numeric columns after binning
def remove_numeric_columns(data):
    for row in data:
        del row["grade_point_average"]
        del row["aptitude_score"]
        del row["communication_score"]
        del row["internship_quality_score"]
    
    return data

# Training the Naive Bayes Model

import random

FEATURES = [
    "backlogs",
    "college_category",
    "country_of_residence",
    "university_ranking",
    "internship_count",
    "specialization",
    "industry",
    "gpa_bin",
    "aptitude_bin",
    "communication_bin",
    "internship_quality_bin"
]

TARGET = "placement_status"
CLASSES = ["Placed", "Not Placed"]

# split the dataset into training and testing sets (80% training, 20% testing)
def train_test_split(data):
    data_copy = data[:]
    random.shuffle(data_copy)
    test_ratio = 0.2
    
    split_index = int(len(data_copy) * (1 - test_ratio))
    
    train_data = data_copy[:split_index]
    test_data = data_copy[split_index:]
    
    # save the test data to a file for later evaluation
    with open("out/test_data.csv", "w") as file:
        headers = data[0].keys()
        file.write(",".join(headers) + "\n")
        
        for row in test_data:
            values = [str(row[header]) for header in headers]
            file.write(",".join(values) + "\n")
            
    with open("out/train_data.csv", "w") as file:
        headers = data[0].keys()
        file.write(",".join(headers) + "\n")
        
        for row in train_data:
            values = [str(row[header]) for header in headers]
            file.write(",".join(values) + "\n")
    
    return train_data, test_data

# Get unique values for each feature
def get_feature_values(data):
    feature_values = {}

    for feature in FEATURES:
        feature_values[feature] = []

    for row in data:
        for feature in FEATURES:
            value = row[feature]
            if value not in feature_values[feature]:
                feature_values[feature].append(value)

    return feature_values

# Compute prior probabilities for each class
def compute_priors(data):
    placed_count = 0
    not_placed_count = 0
    total = len(data)

    for row in data:
        if row[TARGET] == "Placed":
            placed_count += 1
        else:
            not_placed_count += 1

    priors = {
        "Placed": placed_count / total,
        "Not Placed": not_placed_count / total
    }

    return priors

# Count feature values for each class
def count_feature_values(data):
    counts = {
        "Placed": {},
        "Not Placed": {}
    }

    for class_name in CLASSES:
        for feature in FEATURES:
            counts[class_name][feature] = {}

    for row in data:
        class_name = row[TARGET]

        for feature in FEATURES:
            value = row[feature]

            if value not in counts[class_name][feature]:
                counts[class_name][feature][value] = 0

            counts[class_name][feature][value] += 1

    return counts

# Compute likelihoods for each feature value given the class
def compute_likelihoods(data):
    counts = count_feature_values(data)
    feature_values = get_feature_values(data)

    placed_total = 0
    not_placed_total = 0

    for row in data:
        if row[TARGET] == "Placed":
            placed_total += 1
        else:
            not_placed_total += 1

    likelihoods = {
        "Placed": {},
        "Not Placed": {}
    }

    for feature in FEATURES:
        likelihoods["Placed"][feature] = {}
        likelihoods["Not Placed"][feature] = {}

        num_values = len(feature_values[feature])

        for value in feature_values[feature]:
            placed_count = 0
            not_placed_count = 0

            if value in counts["Placed"][feature]:
                placed_count = counts["Placed"][feature][value]

            if value in counts["Not Placed"][feature]:
                not_placed_count = counts["Not Placed"][feature][value]

            # Laplace smoothing
            likelihoods["Placed"][feature][value] = (placed_count + 1) / (placed_total + num_values)
            likelihoods["Not Placed"][feature][value] = (not_placed_count + 1) / (not_placed_total + num_values)

    return likelihoods, feature_values

# Compute feature influence scores based on likelihood differences
def compute_feature_influence(likelihoods, feature_values):
    influence_scores = {}

    for feature in FEATURES:
        total_diff = 0

        for value in feature_values[feature]:
            p_placed = likelihoods["Placed"][feature].get(value, 0)
            p_not_placed = likelihoods["Not Placed"][feature].get(value, 0)

            diff = abs(p_placed - p_not_placed)
            total_diff += diff

        influence_scores[feature] = total_diff

    return influence_scores


def save_feature_influence(influence_scores, filename="out/feature_influence.csv"):
    with open(filename, "w") as file:
        file.write("feature,influence_score\n")

        for feature, score in influence_scores.items():
            file.write(f"{feature},{score}\n")


# Predict the class for a single row of data
def predict_one(row, priors, likelihoods, feature_values, train_data):
    placed_total = 0
    not_placed_total = 0

    for train_row in train_data:
        if train_row[TARGET] == "Placed":
            placed_total += 1
        else:
            not_placed_total += 1

    placed_score = priors["Placed"]
    not_placed_score = priors["Not Placed"]

    for feature in FEATURES:
        value = row[feature]

        if value in likelihoods["Placed"][feature]:
            placed_prob = likelihoods["Placed"][feature][value]
        else:
            placed_prob = 1 / (placed_total + len(feature_values[feature]))

        if value in likelihoods["Not Placed"][feature]:
            not_placed_prob = likelihoods["Not Placed"][feature][value]
        else:
            not_placed_prob = 1 / (not_placed_total + len(feature_values[feature]))

        placed_score = placed_score * placed_prob
        not_placed_score = not_placed_score * not_placed_prob

    if placed_score > not_placed_score:
        return "Placed"
    else:
        return "Not Placed"
    
# Predict the class for all rows in the test dataset
def predict_all(test_data, priors, likelihoods, feature_values, train_data):
    predictions = []

    for row in test_data:
        prediction = predict_one(row, priors, likelihoods, feature_values, train_data)
        predictions.append(prediction)

    return predictions


# Evaluate the model's accuracy
def calculate_accuracy(test_data, predictions):
    correct = 0
    total = len(test_data)

    for i in range(total):
        actual = test_data[i][TARGET]
        predicted = predictions[i]

        if actual == predicted:
            correct += 1

    accuracy = correct / total
    return accuracy

# Save predictions to a CSV file
def save_predictions(test_data, predictions):
    filename="out/predictions.csv"
    with open(filename, "w") as file:
        headers = list(test_data[0].keys()) + ["predicted_placement_status"]
        file.write(",".join(headers) + "\n")

        for i in range(len(test_data)):
            row = test_data[i]
            predicted = predictions[i]
            values = [str(row[header]) for header in test_data[0].keys()] + [predicted]
            file.write(",".join(values) + "\n")
                
# Confusion Matrix and Visualization 
def confusion_matrix(test_data, predictions):
    TP = FP = TN = FN = 0

    for i in range(len(test_data)):
        actual = test_data[i]["placement_status"]
        predicted = predictions[i]

        if actual == "Placed" and predicted == "Placed":
            TP += 1
        elif actual == "Placed" and predicted == "Not Placed":
            FN += 1
        elif actual == "Not Placed" and predicted == "Placed":
            FP += 1
        elif actual == "Not Placed" and predicted == "Not Placed":
            TN += 1

    return TP, FP, TN, FN


import matplotlib.pyplot as plt
# Bar chart visualization of the confusion matrix breakdown
def plot_confusion_bar(TP, FP, TN, FN):
    labels = ["TP", "FP", "FN", "TN"]
    values = [TP, FP, FN, TN]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    plt.title("Confusion Matrix Breakdown")
    plt.xlabel("Prediction Type")
    plt.ylabel("Count")

    for i in range(len(values)):
        ax.text(i, values[i], str(values[i]), ha='center')

    save_path = "out/confusion_bar.png"
    plt.savefig(save_path)
    plt.show()
  
def plot_feature_influence(influence_scores):
    sorted_items = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)

    features = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots()

    ax.bar(features, scores)

    plt.title("Feature Influence")
    plt.xlabel("Features")
    plt.ylabel("Influence Score")

    plt.xticks(rotation=45)

    for i in range(len(scores)):
        ax.text(i, scores[i], f"{scores[i]:.2f}", ha='center')
        
    plt.savefig("out/feature_influence.png")