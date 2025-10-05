from misc import load_data, repeated_evaluate
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def model_factory():
    return DecisionTreeRegressor(random_state=0)

def main():
    df = load_data()
    X = df.drop(columns=['MEDV']).values
    y = df['MEDV'].values
    avg_mse, mses = repeated_evaluate(model_factory, X, y, runs=5, test_size=0.2, seeds=[0,1,2,3,4])
    print("DecisionTreeRegressor average test MSE over 5 runs: {:.4f}".format(avg_mse))
    print("Individual MSEs:", ["{:.4f}".format(m) for m in mses])

if __name__ == "__main__":
    main()