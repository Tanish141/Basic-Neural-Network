# 🧠 XOR Neural Network
This project demonstrates the implementation of a simple Neural Network from scratch in Python to solve the XOR problem, a classic problem in artificial intelligence. The network is trained using the backpropagation algorithm and mean squared error (MSE) as the loss function.

---

### 🚀 Features
Activation Function: Sigmoid function for non-linear decision-making.
Backpropagation: Gradients computed manually to update weights and biases.
Trainable Architecture: Adjustable input, hidden, and output layers.
Loss Function: Mean Squared Error (MSE) for error computation.
Data: Solves the XOR problem (non-linearly separable dataset).

---

### 📂 Dataset
The XOR dataset:

Input	Output
[0, 0]	[0]
[0, 1]	[1]
[1, 0]	[1]
[1, 1]	[0]

---

### ⚙️ How It Works
Initialization: Random initialization of weights and biases.
Forward Propagation: Computes activations through the input, hidden, and output layers using the sigmoid function.
Backward Propagation: Updates weights and biases based on the computed gradients.
Training: Iteratively minimizes the loss using the backpropagation algorithm over multiple epochs.

---

### 📊 Results
The trained Neural Network learns to approximate the XOR logic:

Input	Predicted Output	Actual Output
[0, 0]	~0	0
[0, 1]	~1	1
[1, 0]	~1	1
[1, 1]	~0	0

---

### 🛠️ Technologies Used
Python 🐍
NumPy: For matrix operations and numerical computations.

---

### 🚀 Getting Started
Follow these steps to run the project:

Clone the repository:

bash
Copy code
git clone https://github.com/Tanish141/xor-neural-network.git  
cd xor-neural-network  
Install dependencies:
Ensure you have Python installed. Then, install NumPy if not already present:

bash
Copy code
pip install numpy  
Run the code:
Execute the script in your terminal:

bash
Copy code
python xor_nn.py  
Test the model:
After training, the network predicts XOR outputs based on the input.

---

### 🤝 Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

---

### 📧 Contact
For any queries or suggestions, feel free to reach out:
Email: mrtanish14@gmail.com
GitHub: https://github.com/Tanish141

---

### 🎉 Happy Coding!
