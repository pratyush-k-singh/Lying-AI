from trainer import Trainer, setup_logging, get_log_file_name
import os

if __name__ == "__main__":
    num_episodes = int(input("Enter number of development cycles: "))
    load_previous = input("Load from a previous model? (Y/N): ").lower() == 'y'

    log_file = get_log_file_name()
    setup_logging(log_file)
    print(f"Logging to {log_file}")

    trainer = Trainer(num_episodes=num_episodes)
    
    if load_previous:
        model_dir = '../models'
        model_file = input(f"Enter the model file name from {model_dir} (without extension): ")
        model_path = os.path.join(model_dir, model_file + '.model')
        
        if not os.path.isfile(model_path):
            print(f"Model {model_file}.model not found in {model_dir}. Exiting.")
        else:
            trainer.train(load_model_path=model_path)
    else:
        trainer.train()

    trainer.save_model()
