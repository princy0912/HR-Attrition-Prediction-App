import joblib

# Load the trained model
model = joblib.load("attrition_model.pkl")

# Input employee details (make sure the inputs match training features)
employee = {
    'Age': 35,
    'BusinessTravel': 1,
    'DailyRate': 800,
    'Department': 1,
    'DistanceFromHome': 5,
    'Education': 3,
    'EducationField': 1,
    'EnvironmentSatisfaction': 3,
    'Gender': 1,
    'HourlyRate': 70,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobRole': 4,
    'JobSatisfaction': 3,
    'MaritalStatus': 1,
    'MonthlyIncome': 5000,
    'MonthlyRate': 20000,
    'NumCompaniesWorked': 2,
    'OverTime': 1,
    'PercentSalaryHike': 13,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 2,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 5,
    'YearsInCurrentRole': 2,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 2
}

# Convert to list in right order
input_features = [employee[col] for col in model.feature_names_in_]

# Predict
prediction = model.predict([input_features])[0]
print(f"Prediction (1 = Attrition, 0 = No Attrition): {prediction}")