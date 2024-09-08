# TCR Epitope Classifier

This repository contains a PyTorch implementation of a Transformer-based classifier for T-cell receptor (TCR) epitope prediction.

## Project Structure

```
tcr-epitope-classifier/
│
├── data/
│   └── training_subset_with_classes.csv
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── requirements.txt
├── README.md
└── main.py
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/tcr-epitope-classifier.git
   cd tcr-epitope-classifier
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model and make predictions, run:

```
python main.py
```

## File Descriptions

- `main.py`: The entry point of the program. It orchestrates the data loading, model training, and prediction processes.
- `src/data_processing.py`: Contains functions for loading and preprocessing the TCR data.
- `src/model.py`: Defines the TCRClassifier and TransformerEncoderLayer classes.
- `src/train.py`: Includes the training loop and evaluation function.
- `src/utils.py`: Contains utility functions like the ESM2 embedding function.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
