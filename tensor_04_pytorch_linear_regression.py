import torch 
import matplotlib.pyplot as plt

# Create *known* parameters to use in linear regression Formula Y = a + bX; where a = bias , b = weight
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + weight * X

# Create train/test split

train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test) # 40 40 10 10

def plot_predictions(
                train_data_x=X_train, 
                train_data_y=y_train, 
                test_data_x=X_test, 
                test_data_y=y_test, 
                predictions=None
                ):
  
    """
    Plots training data, test data and compares predictions.
    """

    plt.figure(figsize=(10, 7))       # Makes a blank plotting area => 10 * 7 inches

    # Plot training data in blue
    plt.xlabel("X-axis")
    plt.scatter(train_data_x, train_data_y, c="b", s=4, label="Training data")
    
    # Plot test data in green
    plt.ylabel("Y-axis")
    plt.scatter(test_data_x, test_data_y, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data_x, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


if __name__ == "__main__":
    plot_predictions()
    plt.show()