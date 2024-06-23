import torch 
from torch import nn
import pandas
from sklearn.preprocessing import StandardScaler

def fit_scaler():
    global scaler
    scaler.fit(data.drop(columns=['Risk']))


class HypertensionRiskModel(nn.Module):
    def __init__(self, input_dim):
        super(HypertensionRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 56)
        self.fc5 = nn.Linear(56, 48)
        self.fc6 = nn.Linear(48, 32)
        self.fc7 = nn.Linear(32, 1)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.silu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.silu(x)
        x = self.fc6(x)
        x = self.silu(x)
        x = self.fc7(x)
        x = self.sigmoid(x)  # For binary classification, sigmoid at the end
        return x


# Load the saved model state_dict
model = HypertensionRiskModel(input_dim=12)
model.load_state_dict(torch.load("./hypertension_risk_model.pth"))
model.eval()

data = pandas.read_csv("Hypertension-risk-model-main.csv")
data = data.dropna()
data = data[data['Risk'].isin([0, 1])]
scaler = StandardScaler()

def classify_risk(features):
    features = scaler.transform([features])
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(features)
        risk = "at risk" if prediction > 0.5 else "not at risk"
        print(f"The individual is {risk}.")

def is_valid_number(input_str):
    try:
        float(input_str)
        return True
    except ValueError:
        return False
    
def get_valid_number(prompt):
    user_input = input(prompt)
    if is_valid_number(user_input):
        return float(user_input)
    else:
        print("Invalid input. Please enter a valid number.")
        get_valid_number(prompt)


def get_user_input():
    male = get_valid_number("Enter gender (1 for male, 0 for female): ")
    age = get_valid_number("Enter age: ")
    currentSmoker = get_valid_number("Are you a current smoker? (1 for yes, 0 for no): ")
    cigsPerDay = get_valid_number("Enter number of cigarettes smoked per day: ")
    BPMeds = get_valid_number("Are you on blood pressure medication? (1 for yes, 0 for no): ")
    diabetes = get_valid_number("Do you have diabetes? (1 for yes, 0 for no): ")
    totChol = get_valid_number("Enter total cholesterol level: ")
    sysBP = get_valid_number("Enter systolic blood pressure: ")
    diaBP = get_valid_number("Enter diastolic blood pressure: ")
    BMI = get_valid_number("Enter body mass index (BMI): ")
    heartRate = get_valid_number("Enter heart rate: ")
    glucose = get_valid_number("Enter glucose level: ")
    
    return [male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]

def chatbot():
    print("Hello! I am your health assistant. I can help you check if you need to change your blood pressure medication.")
    user_input = input("Please type 's' to get started or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Goodbye! Stay healthy!")
        return
    features = get_user_input()
    response = classify_risk(features)
    print(response)

fit_scaler()
chatbot()