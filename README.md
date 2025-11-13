\# Cifar-10-Classifier



This repository contains my implementation of a \*\*CIFAR-10 image classification project\*\*, originally developed as part of a machine learning assignment (COMP5318, University of Sydney).



The project compares multiple models on the CIFAR-10 dataset, including:



\- Traditional machine learning models (Random Forest)

\- Shallow feed-forward neural networks (MLP)

\- Convolutional Neural Networks (CNN)



The repository is structured for clarity, modularity, and reproducibility.



---



\## 📁 Project Structure



Cifar-10-Classifier/

├── README.md

├── LICENSE

├── requirements.txt

├── .gitignore

├── src/

│ ├── dataset.py

│ ├── models.py

│ ├── train.py

│ └── evaluate.py

├── notebooks/

│ └── COMP5318-assignment2-template-notebook.ipynb

├── results/

└── data/



\## ▶️ Getting Started



\### 1. Clone the repository



git clone https://github.com/<your-username>/Cifar-10-Classifier.git

cd Cifar-10-Classifier



\### 2. Install dependencies



pip install -r requirements.txt





\### 3. Run training



python src/train.py --epochs 10 --batch-size 128 --lr 0.001





All saved models, logs and metrics will appear under the `results/` directory.



---



\## 📓 Notebook Version



The original assignment notebook can be found in:



notebooks/COMP5318-assignment2-template-notebook.ipynb





You can run it via:



jupyter notebook



---



\## 💡 Notes



\- The `src/` directory contains modularized Python scripts  

\- You can expand this project with new models anytime  

\- Ideal for GitHub portfolio / resume showcase



---



\## 🧾 License



Released under the MIT License. See `LICENSE` for details.

