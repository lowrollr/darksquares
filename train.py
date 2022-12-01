from training.training_bot import start_training
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    start_training(batch_size=20)