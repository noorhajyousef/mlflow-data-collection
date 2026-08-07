import mlflow

mlflow.set_experiment("demo-experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.93)
    mlflow.sklearn.log_model(model, "model")
