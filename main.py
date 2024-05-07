import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Function to generate fixed grades for a specified number of semesters
def generate_student_grades_fixed(num_students, num_semesters, include_grad_year=False):
    student_data = []
    for _ in range(num_students):
        grades = np.round(np.random.uniform(1.0, 4.0, num_semesters), 2)
        student = {"grades": list(grades)}
        if include_grad_year:
            grad_year = np.round(np.random.uniform(4.0, 7.0), 1)
            student["graduation_year"] = grad_year
        student_data.append(student)
    return pd.DataFrame(student_data)

# Generate data for 100 students
data_fixed_8_semesters = generate_student_grades_fixed(100, 8, include_grad_year=True)

# Prepare the data by converting grades list into separate columns
def prepare_data(df):
    grades_df = pd.DataFrame(df['grades'].tolist(), columns=[f'Semester_{i + 1}' for i in range(8)])
    grades_df['Graduation_Year'] = df['graduation_year']
    # Categorizing graduation time
    grades_df['Grad_Above_5_Years'] = (grades_df['Graduation_Year'] > 5).astype(int)
    return grades_df

data = prepare_data(data_fixed_8_semesters)

# Feature engineering
def feature_engineering(df):
    df['Grade_Semester_5'] = df['Semester_5']  # Directly using the grade of semester 5
    return df

data = feature_engineering(data)

# Feature selection focusing on Semester 5 as an influenced grade, since it is the most important feature
X = data[['Grade_Semester_5']]
y = data['Grad_Above_5_Years']

# Modeling with Gaussian Naive Bayes
kf = KFold(n_splits=100, shuffle=True, random_state=42)
model = GaussianNB()
accuracy_list = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_list.append(accuracy)

average_accuracy = np.mean(accuracy_list)
print(f"Average Accuracy: {average_accuracy*100:.2f}%")
