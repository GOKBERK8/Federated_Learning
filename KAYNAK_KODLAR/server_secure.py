import flwr as fl
import numpy as np

# Tunahan Hoca'nın önerisi üzerine geliştirilen Dinamik Güven Skoru Stratejisi
class DynamicTrustStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # 1. İstemcilerden gelen ağırlıkları ve veri sayılarını al
        client_weights = []
        client_sizes = []
        for _, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(weights)
            client_sizes.append(fit_res.num_examples)

        # 2. Referans (Güvenli) bir merkez noktası bulmak için medyan alıyoruz
        # Bu, kötü niyetli veriden etkilenmeyen "ideal" ağırlık noktasıdır
        reference_weights = [
            np.median(np.array([client_weights[i][layer] for i in range(len(client_weights))]), axis=0)
            for layer in range(len(client_weights[0]))
        ]

        trusted_sizes = []
        print(f"\n--- TUR {server_round}: DİNAMİK GÜVEN SKORU ANALİZİ ---")
        
        # 3. Her istemcinin güven skorunu hesapla
        for i, weights in enumerate(client_weights):
            # İstemcinin ağırlıkları ile referans noktası arasındaki farkı (mesafeyi) ölç
            distance = sum(np.linalg.norm(weights[layer] - reference_weights[layer]) for layer in range(len(weights)))
            
            # Güven Skoru Çarpanı: Mesafe ne kadar büyükse, güven o kadar sıfıra yaklaşır
            # distance + 1e-5 ile sıfıra bölünme hatasını engelliyoruz
            trust_score = 1.0 / (1.0 + distance)
            
            # Hastanenin orijinal veri sayısını (oy gücünü) güven skoruyla çarpıyoruz
            # HOCANIN İSTEDİĞİ: Veriyi çöpe atma, sadece etkisini azalt!
            dynamic_weight = client_sizes[i] * trust_score
            trusted_sizes.append(dynamic_weight)
            
            print(f"İstemci {i+1} -> Sapma Mesafesi: {distance:.4f} | Güven Skoru (Çarpan): {trust_score:.4f}")

        # 4. Modifiye edilmiş (güvenlikten geçmiş) ağırlıklarla standart birleştirme yap
        # Parametreleri orijinal Flower formatına geri dönüştür
        new_results = []
        for i, (client, fit_res) in enumerate(results):
            # Sadece num_examples (etki gücü) değerini yeni hesapladığımız dynamic_weight ile değiştiriyoruz
            fit_res.num_examples = trusted_sizes[i]
            new_results.append((client, fit_res))

        # Ana sınıfın (FedAvg) birleştirme fonksiyonunu yeni güvenli değerlerle çağır
        return super().aggregate_fit(server_round, new_results, failures)

def main():
    print("KALKAN AKTİF: Dinamik Güven Skoru (Dynamic Trust) destekli sunucu başlatılıyor...")
    
    # Kendi yazdığımız stratejiyi kullanıyoruz
    strategy = DynamicTrustStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()