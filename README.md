

# Self-Organizing Map (SOM)

This program implements a Self-Organizing Map (SOM) for data clustering and visualization.
It comes with the file "data.csv" to be used as an example of how cluster customers based on features from a CSV file, such as income and spending score. It visualizes the clustering results in an image file, useful for customer segmentation analysis.

### Features
- **Flexible Input**: Reads data from a CSV file, allowing the selection of specific columns for clustering.
- **Customizable SOM Parameters**: Configure SOM grid dimensions and iteration counts via `config.yaml`.
- **Image Output**: Saves the trained SOM map as a visual PNG file.

### Requirements
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `pandas`, `pyyaml`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the program using the following command:
```bash
python execSOM.py data.csv "Age" "Annual_Income" "Spending_Score" 
```

- `data.csv`: Path to the CSV file containing customer data.
- `"Age"`: Column name for the age of customers.
- `"Annual_Income"`: Column name for the annual income of customers.
- `"Spending_Score"`: Column name for the customer spending score.

### Example CSV File Format

```csv
CustomerID,Gender,Age,Annual_Income,Spending_Score
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
...
```

### Configuration

Modify `config.yaml` to customize SOM grid dimensions, number of training iterations, and image output paths.


