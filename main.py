# main.py for Job Placement Dataset Analysis using Naive Bayes Algorithm

import functions

dataset = functions.load_data('data/JOB_PLACEMENT_DATASET.csv')
data = functions.convert_numeric(dataset)
data = functions.clean_categorical(dataset)
data = functions.apply_binning(dataset)
data = functions.remove_numeric_columns(dataset)


train_data, test_data = functions.train_test_split(data)
priors = functions.compute_priors(train_data)
likelihoods, feature_values = functions.compute_likelihoods(train_data)
predictions = functions.predict_all(test_data, priors, likelihoods, feature_values, train_data)


functions.save_predictions(test_data, predictions)
accuracy = functions.calculate_accuracy(test_data, predictions)


for i in range(5):
    print(f"[{i+1}] - Backlogs: {test_data[i]['backlogs']} \n" 
          f" College Category: {test_data[i]['college_category']} \n"
          f" Country of Residence: {test_data[i]['country_of_residence']} \n"
          f" University Ranking: {test_data[i]['university_ranking']} \n"
          f" Internship Count: {test_data[i]['internship_count']} \n"
          f" Specialization: {test_data[i]['specialization']} \n"
          f" Industry: {test_data[i]['industry']} \n"
          f" GPA: {test_data[i]['gpa_bin']} \n"
          f" Aptitude: {test_data[i]['aptitude_bin']} \n"
          f" Communication: {test_data[i]['communication_bin']} \n"
          f" Internship Quality: {test_data[i]['internship_quality_bin']} \n"
          f" Actual: {test_data[i]['placement_status']} || Predicted: {predictions[i]}")
    if test_data[i]['placement_status'] == predictions[i]:
        print(">> Matched\n")
    else:
        print(">> Not Matched\n")
    
    
TP, FP, TN, FN = functions.confusion_matrix(test_data, predictions)

print("\nConfusion Matrix:")
print("True Positive:", TP, "\nFalse Positive:", FP)
print("\nFalse Negative:", FN, "\nTrue Negative:", TN)

print(f"\nAccuracy: {accuracy:.2f}%")

functions.plot_confusion_bar(TP, FP, TN, FN)

