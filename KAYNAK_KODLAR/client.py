import pandas as pd
import flwr as fl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import sys
import numpy as np
import warnings

# Gereksiz terminal uyarılarını gizliyoruz
warnings.simplefilter("ignore")

client_id = sys.argv[1]
print(f"İstemci {client_id} başlatılıyor...")

df = pd.read_csv(f"client_{client_id}_data.csv")
X = df.drop(columns=['target']).values
y_numeric = df['target'].values

# Sınıflandırmaya çevirme
y = np.where(y_numeric > 140, 1, 0)

# Model: SGDClassifier
model = SGDClassifier(max_iter=1000, random_state=42, warm_start=True)
model.fit(X, y) # Modeli başlatmak için ilk uyandırma

class DiyabetClient(fl.client.NumPyClient):
    
    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

    def get_parameters(self, config):
        return [model.coef_, model.intercept_]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(X, y)
        print(f"--- İstemci {client_id}: Model Eğitildi ---")
        return self.get_parameters(config), len(X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = model.predict(X)
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        print(f"--- İstemci {client_id}: Test Edildi -> Accuracy: {acc:.4f}, F1 Skoru: {f1:.4f} ---")
        # Flower ana loss değeri olarak float ister, biz (1 - F1) değerini hata oranı gibi yolluyoruz
        return float(1.0 - f1), len(X), {"accuracy": acc, "f1": f1}

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=DiyabetClient().to_client(),
)