import pandas as pd
import flwr as fl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import sys
import numpy as np
import warnings

warnings.simplefilter("ignore")

client_id = sys.argv[1]
print(f"KÖTÜ NİYETLİ İstemci {client_id} başlatılıyor...")

df = pd.read_csv(f"client_{client_id}_data.csv")
X = df.drop(columns=['target']).values
y_numeric = df['target'].values

y = np.where(y_numeric > 140, 1, 0)

# --- STRIDE: TAMPERING (LABEL FLIPPING SALDIRISI) ---
# Hastaları sağlıklı, sağlıklıları hasta yapıyoruz (1-y mantığı 1'leri 0, 0'ları 1 yapar)
print("DİKKAT: Veri zehirleniyor (Etiketler Tersine Çevriliyor!)...")
y = 1 - y  
# ---------------------------------------------------

model = SGDClassifier(max_iter=1000, random_state=42, warm_start=True)
model.fit(X, y)

class MaliciousDiyabetClient(fl.client.NumPyClient):
    
    def set_parameters(self, parameters):
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

    def get_parameters(self, config):
        return [model.coef_, model.intercept_]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(X, y)
        print(f"--- KÖTÜ NİYETLİ İstemci {client_id}: ZEHİRLİ Model Eğitildi ---")
        return self.get_parameters(config), len(X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)
        print(f"--- KÖTÜ NİYETLİ İstemci {client_id} Testi -> Accuracy: {acc:.4f}, F1 Skoru: {f1:.4f} ---")
        return float(1.0 - f1), len(X), {"accuracy": acc, "f1": f1}

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=MaliciousDiyabetClient().to_client(),
)