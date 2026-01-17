from ai_services.consciousness.grid_manager import grid_manager
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def seed_test_data():
    print("🧠 Seeding Mostar Grid with test detections...")
    
    test_detections = [
        {
            "species": "Mastomys_natalensis",
            "confidence": 0.95,
            "latitude": 6.5,
            "longitude": 7.3,
            "odu": "Eji_Ogbe",
            "risk": 0.945
        },
        {
            "species": "Mastomys_erythroleucus",
            "confidence": 0.88,
            "latitude": 7.2,
            "longitude": 3.9,
            "odu": "Iwori_Meji",
            "risk": 0.504
        }
    ]

    for det in test_detections:
        detection_data = {
            "detections": [{"species": det["species"], "confidence": det["confidence"]}],
            "location": {"latitude": det["latitude"], "longitude": det["longitude"]}
        }
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "odu_pattern": det["odu"],
            "risk_assessment": {"score": det["risk"]},
            "ubuntu_guidance": "Community vigilance is key."
        }
        
        print(f"-> Persisting {det['species']} at ({det['latitude']}, {det['longitude']})")
        grid_manager.persist_detection(detection_data, analysis)

    print("✅ Seeding complete. Check Neo4j Browser!")

if __name__ == "__main__":
    seed_test_data()
