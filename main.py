import logging
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
from som import SelfOrganizingMap
from utils import load_data_from_csv, load_config

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_trained_image(trained_weights, filename):
    """
    Save the trained weights of the SOM as an image.

    Parameters:
    - trained_weights: np.ndarray
        The matrix of trained weights to be saved as an image.
    - filename: str
        The filename for the saved image.
    """
    plt.imsave(filename, trained_weights)  # Save the image with trained SOM weights.


def main(data_file, data_columns):
    """
    Main function to run the SOM training process and save the output image.

    Parameters:
    - data_file: str
        The path to the CSV file containing input data.
    - data_columns: list of str
        The list of columns to be used as input features for the SOM.
    """
    try:
        # Load configuration from the config.yaml file
        logging.info("Loading configuration...")
        config = load_config()

        # Load data from the specified CSV file and columns
        logging.info("Loading data from CSV...")
        input_data = load_data_from_csv(data_file, data_columns)
        
        # Retrieve grid size and iteration parameters from the configuration
        grid_width = config['som']['grid_width']
        grid_height = config['som']['grid_height']
        max_iterations = config['som']['max_iterations']

        # Initialize the SOM with the parameters from the configuration
        logging.info(f"Initializing SOM with grid size {grid_width}x{grid_height} and {max_iterations} iterations.")
        som = SelfOrganizingMap(grid_width, grid_height, max_iterations)

        # Train the SOM and measure the time taken
        start_time = time.time()
        trained_weights = som.train(input_data)
        elapsed_time = time.time() - start_time
        logging.info(f"SOM training completed in {elapsed_time:.2f} seconds.")

        # Save the trained SOM weights as an image
        output_file = "som_result.png"
        logging.info(f"Saving the resulting image to {output_file}")
        save_trained_image(trained_weights, output_file)

        logging.info("SOM process complete.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run SOM on customer data and generate a clustering image.")
    
    # Argument for CSV file path
    parser.add_argument('data_file', type=str, help="Path to the input CSV file containing customer data.")
    
    # Argument for feature columns to use in SOM
    parser.add_argument('data_columns', type=str, nargs='+', help="List of columns to use as input features for the SOM.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the main function with the provided file name and column names
    main(args.data_file, args.data_columns)
