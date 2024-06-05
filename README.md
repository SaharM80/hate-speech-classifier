# Hate Speech Classification

This project focuses on classifying text into different categories of hate speech using machine learning techniques. Hate speech classification has various use cases across different domains, including social media moderation, content filtering, and sentiment analysis. By accurately identifying hate speech in text data, this classifier can contribute to creating safer online environments, promoting respectful discourse, and preventing the spread of harmful content.

## Installation

This project requires the installation of the following dependencies:
- `datasets`
- `transformers`

These dependencies can be installed via pip:
```
!pip install datasets
!pip install transformers
```

## Usage

1. **Downloading and Saving the Dataset**: The code downloads the hate speech dataset and saves it to a specified directory.

2. **Preprocessing**: The dataset is loaded and preprocessed. The labels are separated from the text data.

3. **Tokenizing**: The text data is tokenized using the RoBERTa tokenizer.

4. **Splitting the Dataset**: The dataset is split into training and testing sets.

5. **Training and Saving the Model**: The model is trained using the training dataset. Training arguments such as batch size, learning rate, and number of epochs can be modified as required. The trained model is saved for later use.

6. **Loading and Testing the Model**: The trained model is loaded and tested with sample text inputs to predict the labels.

## File Structure

- `README.md`: This file contains information about the project, its usage, and dependencies.
- `hate_speech_classification.ipynb`: Jupyter Notebook containing the code for hate speech classification.
- `models/`: Directory to store trained models.
- `data/`: Directory to store dataset files.

## How to Use

1. Clone this repository.
2. Run the Jupyter Notebook `hate_speech_classification.ipynb`.
3. Modify the code as needed for your dataset or model architecture.
4. Train the model and save it for inference.
5. Use the trained model for hate speech classification in your applications.

## Credits

- This project utilizes the Hugging Face `datasets` and `transformers` libraries for dataset handling and model training.
- The hate speech dataset used in this project is sourced from the "hate_speech18" dataset available through Hugging Face's `datasets`.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
