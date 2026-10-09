import flwr as fl

def main():
    # FedAvg (Federated Averaging - Federe Ortalama) stratejisi
    # Bu algoritma, istemcilerden gelen ağırlıkların ortalamasını alarak modeli günceller.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,            # Eğitime bağlanacak istemci oranı (%100)
        fraction_evaluate=1.0,       # Teste bağlanacak istemci oranı (%100)
        min_fit_clients=3,           # Eğitimin başlaması için gereken minimum istemci (3 hastanemiz var)
        min_evaluate_clients=3,      # Test için gereken minimum istemci
        min_available_clients=3,     # Sistemin başlaması için beklemesi gereken toplam istemci sayısı
    )

    print("Merkezi sunucu (Server) başlatılıyor. İstemciler bekleniyor...")
    
    # Sunucuyu 8080 portundan başlatıyoruz.
    # num_rounds=3 demek, istemcilerle 3 tur (round) haberleşip eğitimi tamamlayacak demektir.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()