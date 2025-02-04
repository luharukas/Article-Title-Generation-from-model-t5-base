# Article-Title-Generation-from-model-t5-base

This repository contains code for the "Automatic Article Title Generation" project. The project aims to generate article titles automatically using the T5 base model. It provides scripts for data preprocessing, model training, and title generation.

## Project Structure

The project is structured as follows:

- `train.py`: This script handles the training of the Seq2Seq model. It preprocesses the training data, prepares it for training, loads the pre-trained T5 model, and trains the model using the defined training arguments. The trained model is then saved for future use.

- `test.py`: This script demonstrates the title generation process. It loads the trained model and tokenizer, applies the model to generate titles for a list of texts, evaluates the generated titles using the ROUGE score, and saves the results in a CSV file.

- `function.py`: This file contains various functions used for data preprocessing, model preparation, and evaluation.

- `configuration.py`: This file contains configuration parameters such as maximum text length, maximum title length, batch size, learning rate, and model checkpoint path.

## Dependencies

The project requires the following dependencies:

- pandas
- nltk
- numpy
- rouge_score
- contractions
- datasets
- transformers


To install the dependencies, you can use the following command: `pip install -r requirements.txt`

To train the model, run the following command:  `python train.py`

This script will preprocess the training data, prepare it for training, load the pre-trained T5 model, and start the training process. The trained model will be saved for future use.

Run the following command for generating new titles: `python test.py`


This script will load the trained model and tokenizer, apply the model to generate titles for the new texts, evaluate the generated titles using the ROUGE score, and save the results in the `submission.csv` file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
