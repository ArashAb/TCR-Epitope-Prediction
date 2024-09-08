from src.data_processing import load_data, preprocess_data
from src.model import TCRClassifier
from src.train import train_model
from src.utils import evaluate

def main():
    # Load and preprocess data
    file_path = 'data/training_subset_with_classes.csv'
    data, label_encoders, epitope_encoder = load_data(file_path)
    
    # Initialize model
    vocab_sizes = {col: len(label_encoders[col].classes_) for col in label_encoders}
    model = TCRClassifier(num_classes=len(epitope_encoder.classes_), vocab_sizes=vocab_sizes)
    
    # Train model
    trained_model = train_model(model, data, label_encoders, epitope_encoder)
    
    # Here you could add code to make predictions or further evaluate the model
    
    print("Training completed!")

if __name__ == "__main__":
    main()
