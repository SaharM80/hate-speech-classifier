# Hate Speech Classification

This project focuses on classifying text into two categories of text that qualifies as hate speech and text that doesn't by preprocessing the text and finetuning the data on an already finetuned version of the `RoBERTa` model on `HuggingFace`. Hate speech classification has various use cases across different domains, including social media moderation, content filtering, and sentiment analysis. By accurately identifying hate speech in text data, this classifier can contribute to creating safer online environments, promoting respectful discourse, and preventing the spread of harmful content. 


## Installation

This project requires the installation of the following dependencies that can be installed via pip:
- `datasets`
- `transformers`
- `pandas`
- `numpy`

## Training Steps and Results

1. **Downloading and Saving the Dataset**: The code downloads the hate speech dataset and saves it to a specified directory.

2. **Preprocessing**: The dataset is loaded and preprocessed. The labels are separated from the text data.

3. **Tokenizing**: The model and tokenizer used in this project was the facebook/roberta-hate-speech-dynabench-r4-target model and the text data is tokenized using its tokenizer.

4. **Splitting the Dataset**: The dataset is split into training and testing sets.

5. **Training and Saving the Model**: The model is trained using the training dataset. Training arguments such as batch size, learning rate, and number of epochs can be modified as required. The trained model is saved for later use. </br></br>
   The table below illustrates how my model performed during training over six epochs. It provides a detailed view of the training loss, validation loss, and accuracy for each epoch.
   | Epoch | Training Loss | Validation Loss | Accuracy  |
   |-------|---------------|-----------------|-----------|
   | 1     | 0.399700      | 0.338060        | 0.883965  |
   | 2     | 0.343100      | 0.343596        | 0.891275  |
   | 3     | 0.251200      | 0.374576        | 0.905436  |
   | 4     | 0.206800      | 0.381377        | 0.915487  |
   | 5     | 0.134100      | 0.442791        | 0.921425  |
   | 6     | 0.095000      | 0.467573        | 0.919141  |

7. **Loading and Testing the Model**: The trained model is loaded and tested with a 'test_model' function for testing sample text and predicting their label.

## File Structure

- `README.md`: This file contains information about the project, its usage, and dependencies. It also contains the link to where the trained model is stored.
- `hate_speech_classification.ipynb`: Jupyter Notebook containing the code for training the classifier and inference.
- `models/`: Directory to store trained models.
- `data/`: Directory to store dataset files. The dataset file format is the HuggingFace '.hf' format.

## How to Use
The model was too large for github. here is the link to the trained model on my google colab drive:
<a href="https://drive.google.com/drive/folders/1G2pdxRgoREHW6_jKX5zufle2w70pcdoj?usp=sharing">Model link</a>
</br>
However if you want to train your own model but use my code you can:
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
