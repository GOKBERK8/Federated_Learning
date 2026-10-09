from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_diabetes
import numpy as np

print("Sınıflandırma için Merkezi model verisi hazırlanıyor...")
# Tüm veriyi tekrar yüklüyoruz
diabetes = load_diabetes()
X = diabetes.data
y_numeric = diabetes.target

# REGRESYONU SINIFLANDIRMAYA ÇEVİRME
# 140 puanın üzerindekiler 1 (Yüksek Risk), altındakiler 0 (Düşük Risk)
y = np.where(y_numeric > 140, 1, 0)

# Veriyi Eğitim (%80) ve Test (%20) olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SGDRegressor yerine artık Sınıflandırıcı (SGDClassifier) kullanıyoruz
model = SGDClassifier(max_iter=1000, random_state=42)

print("Merkezi model eğitiliyor...")
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapıp başarıyı ölçüyoruz
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Merkezi Model (Baseline) Başarı Metrikleri ---")
print(f"Doğruluk Oranı (Accuracy): {acc:.4f}")
print(f"F1 Skoru: {f1:.4f}")
print("--------------------------------------------------")
print("Not: F1 Skoru dengesiz veri setlerinde asıl başarıyı gösterir.")