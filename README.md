# HealthcareChatbot
BerkeleyAIHackathon

In the healthcare industry, collecting electronic clinical quality measure (eCQM) data is essential for monitoring and improving patient outcomes. Hospitals typically gather this data to enhance adherence to treatment plans for chronic conditions like hypertension. While patients can easily take blood pressure measurements at home, the subsequent data analysis to determine hypertension risk and medication adjustments can be labor-intensive for healthcare providers.

BP Buddy is our innovative healthcare chatbot designed to streamline this process and mitigate medical burnout by reducing the workload of medical professionals.

BP Buddy queries patients to gather comprehensive data on various hypertension-related factors, including gender, age, smoking status, blood pressure medication usage, diabetes status, total cholesterol, systolic and diastolic blood pressure, BMI, heart rate, and glucose levels. Leveraging a fully connected neural network (FNN) model, BP Buddy analyzes these inputs to predict whether a patient is at risk of hypertension. If a patient is flagged as at risk, their profile in the hospital system is marked for further review by a medical professional, enabling timely medication adjustments and necessary interventions to ensure optimal patient care.

Our model was trained on the Hypertension-risk-model-main.csv dataset from Kaggle, encompassing diverse patient data to enhance predictive accuracy. The neural network architecture features multiple layers, including SiLU and ReLU activation functions, to capture complex patterns in the data. We set the batch size to 4 and the learning rate to 0.0008, optimizing training efficiency and performance. For loss calculation, we employed PyTorch's binary cross-entropy loss function, and the Adam optimizer was utilized for efficient gradient descent.

BP Buddy is poised to revolutionize hypertension management by providing an efficient, accurate, and user-friendly solution for both patients and healthcare providers.