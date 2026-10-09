import pandas as pd
from sklearn.datasets import load_diabetes

# 1. Scikit-learn içindeki diyabet veri setini yüklüyoruz
print("Veri seti yükleniyor...")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# X (özellikler) ve y (hedef) verilerini birleştirip bir Pandas DataFrame oluşturuyoruz
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

# Veriyi rastgele karıştırıyoruz
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Veriyi 3 eşit parçaya bölüyoruz (Saf Pandas yöntemi)
parca_buyuklugu = len(df) // 3

# 3. Her bir parçayı ayrı CSV dosyaları olarak kaydediyoruz
for i in range(3):
    baslangic = i * parca_buyuklugu
    # Son parça kalan tüm veriyi garantiye almak için sona kadar gider
    bitis = (i + 1) * parca_buyuklugu if i < 2 else len(df)
    
    parca = df.iloc[baslangic:bitis]
    dosya_adi = f"client_{i+1}_data.csv"
    parca.to_csv(dosya_adi, index=False)
    print(f"{dosya_adi} oluşturuldu. (Satır sayısı: {len(parca)})")

print("Veri bölme işlemi başarıyla tamamlandı!")