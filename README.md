Agriculture is the backbone of our society, yet millions of crops are lost every year due to undetected plant diseases. AGRO-CARE is an AI-powered plant disease classifier that uses plant leaf images to predict the disease. This project aims to assist farmers and agronomists in early disease detection, reducing crop losses, and promoting sustainable agriculture. This project empowers farmers and agricultural experts to quickly and accurately diagnose plant diseases.



**OBJECTIVES**

Enhance early detection of plant diseases to minimize agricultural losses.

Provide real-time analysis through a user-friendly web-based interface.

Leverage AI and Deep Learning to ensure high accuracy in classification.

Support farmers by reducing reliance on pesticides through precise diagnosis.

**Dataset**

Used the newplantvillage Dataset from kaggle https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset, it contains over 60,000 images of healthy and diseased plant leaves across various species. The dataset is preprocessed and augmented to improve the model's generalization.

**Image Preprocessing**

Resizing & Cropping: Standardized all images to 224x224 pixels.

Data Augmentation: Applied rotation, flipping, and normalization for better model generalization.

**Deep Learning Model**

The classifier uses a CNN architecture with Batch Normalization & Dropout to prevent overfitting.

**Running the Project**

1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Train the Model

python Plant_disease_main.py

3️⃣ Run the FastAPI Server

uvicorn app:app --reload

**Results & Accuracy**

Achieved 95%+ accuracy on the PlantVillage dataset.

Reduced disease detection time to seconds.

Provided a scalable, real-time plant disease detection system.

**Future Enhancements**

Deploy on Cloud (AWS/GCP) for wider accessibility.

Develop a mobile app for farmers to scan leaves instantly.

Integrate IoT sensors for smart farming solutions.

**Contributing**

Contributions are welcome! Feel free to fork the repository and submit a pull request.

**License**

This project is open-source under the MIT License.
