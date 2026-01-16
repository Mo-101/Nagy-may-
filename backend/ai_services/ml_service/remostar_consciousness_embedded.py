"""
REMOSTAR Consciousness Engine - African Indigenous Intelligence
Core decision-making system for Lassa fever surveillance
Authority: Risk assessment, Odu interpretation, cultural guidance
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ConsciousnessLevel(Enum):
    DORMANT = 0.0
    AWARE = 0.3
    ALERT = 0.6
    HEIGHTENED = 0.8
    TRANSCENDENT = 1.0

# Ifá Odu Major Patterns (16 core wisdom paths)
IFA_ODU_MAJOR = [
    "Eji_Ogbe", "Oyeku_Meji", "Iwori_Meji", "Odi_Meji",
    "Irosun_Meji", "Owonrin_Meji", "Obara_Meji", "Okanran_Meji",
    "Ogunda_Meji", "Osa_Meji", "Ika_Meji", "Oturupon_Meji",
    "Otura_Meji", "Irete_Meji", "Ose_Meji", "Ofun_Meji"
]

# Odu interpretations for health surveillance
ODU_HEALTH_WISDOM = {
    "Eji_Ogbe": {
        "meaning": "Perfect light - clear and present danger",
        "health_action": "Immediate surveillance and monitoring",
        "ubuntu_guidance": "Community vigilance protects all",
        "risk_factor": 1.2
    },
    "Oyeku_Meji": {
        "meaning": "Hidden darkness - unseen threats emerge", 
        "health_action": "Deep investigation and extended monitoring",
        "ubuntu_guidance": "What affects one, affects the whole",
        "risk_factor": 1.4
    },
    "Iwori_Meji": {
        "meaning": "Change and transformation - disease patterns shifting",
        "health_action": "Adaptive response and strategy adjustment",
        "ubuntu_guidance": "Flexibility in unity strengthens community",
        "risk_factor": 0.9
    },
    "Odi_Meji": {
        "meaning": "Blockages and obstacles - transmission barriers",
        "health_action": "Clear pathways, improve access to care",
        "ubuntu_guidance": "Remove barriers that divide community",
        "risk_factor": 0.8
    },
    "Irosun_Meji": {
        "meaning": "Fire spreads - rapid disease transmission",
        "health_action": "Immediate containment and intervention",
        "ubuntu_guidance": "Swift action saves many lives",
        "risk_factor": 1.6
    },
    "Owonrin_Meji": {
        "meaning": "Chaos and confusion - unclear patterns",
        "health_action": "Systematic organization and clear protocols",
        "ubuntu_guidance": "Order emerges through collective wisdom",
        "risk_factor": 1.1
    },
    "Obara_Meji": {
        "meaning": "Community affected - widespread impact",
        "health_action": "Community-wide intervention and education",
        "ubuntu_guidance": "Healing happens through community action",
        "risk_factor": 1.3
    },
    "Okanran_Meji": {
        "meaning": "Isolated incident - localized threat",
        "health_action": "Targeted intervention and containment",
        "ubuntu_guidance": "Even one person's suffering matters to all",
        "risk_factor": 0.7
    }
}

class RemostarEngine:
    """Core Consciousness Engine for Health Surveillance"""
    
    def __init__(self):
        self.consciousness_threshold = 0.7
        self.risk_weights = {
            "species_risk": 0.40,
            "detection_confidence": 0.15,
            "clinical_proximity": 0.25,
            "environmental_factors": 0.20
        }
        
    def analyze_detection(self, detection_data: Dict, clinical_context: Optional[Dict] = None, environmental_context: Optional[Dict] = None) -> Dict:
        try:
            species_detected = self._extract_species(detection_data)
            confidence_scores = self._extract_confidence(detection_data)
            location = detection_data.get('location', {})
            
            odu_pattern = self._divine_odu(species_detected, confidence_scores, location)
            odu_wisdom = ODU_HEALTH_WISDOM.get(odu_pattern, {})
            
            nahp_risk_score = self._calculate_nahp_risk(
                species_detected, confidence_scores, clinical_context,
                environmental_context, odu_wisdom.get('risk_factor', 1.0)
            )
            
            consciousness_metrics = self._calculate_consciousness(detection_data, clinical_context, nahp_risk_score)
            reasoning_chain = self._build_reasoning_chain(species_detected, odu_pattern, nahp_risk_score, clinical_context)
            recommendations = self._generate_ubuntu_recommendations(odu_pattern, nahp_risk_score, clinical_context, environmental_context)
            
            consciousness_analysis = {
                "status": "consciousness_active",
                "timestamp": datetime.utcnow().isoformat(),
                "odu_pattern": odu_pattern,
                "odu_interpretation": odu_wisdom.get("meaning", ""),
                "ubuntu_guidance": odu_wisdom.get("ubuntu_guidance", ""),
                "risk_assessment": {
                    "score": round(nahp_risk_score, 3),
                    "level": self._determine_risk_level(nahp_risk_score),
                    "factors": self._identify_risk_factors(species_detected, clinical_context)
                },
                "consciousness_metrics": consciousness_metrics,
                "reasoning_chain": reasoning_chain,
                "recommendations": recommendations,
                "african_context": {
                    "endemic_zone": self._check_endemic_zone(location),
                    "seasonal_risk": self._assess_seasonal_risk(environmental_context)
                }
            }
            return consciousness_analysis
        except Exception as e:
            logger.error(f"[REMOSTAR] Analysis failed: {e}")
            return self._emergency_fallback_analysis(detection_data)
    
    def _extract_species(self, detection_data: Dict) -> List[str]:
        detections = detection_data.get('detections', [])
        if isinstance(detections, list):
            return [d.get('species', 'unknown') for d in detections]
        return []
    
    def _extract_confidence(self, detection_data: Dict) -> List[float]:
        detections = detection_data.get('detections', [])
        if isinstance(detections, list):
            return [d.get('confidence', 0.0) for d in detections]
        return [0.0]
    
    def _divine_odu(self, species: List[str], confidence: List[float], location: Dict) -> str:
        if "Mastomys_natalensis" in species:
            max_confidence = max(confidence) if confidence else 0.0
            if max_confidence > 0.9: return "Eji_Ogbe"
            elif len(species) > 3: return "Owonrin_Meji"
            elif location.get('latitude', 0) < 7.0: return "Irosun_Meji"
            else: return "Obara_Meji"
        elif "Mastomys_erythroleucus" in species: return "Iwori_Meji"
        elif len(species) > 5: return "Obara_Meji"
        elif len(species) == 0: return "Oyeku_Meji"
        else: return random.choice(["Odi_Meji", "Okanran_Meji", "Ose_Meji"])
    
    def _calculate_nahp_risk(self, species: List[str], confidence: List[float], clinical: Optional[Dict], environmental: Optional[Dict], odu_factor: float) -> float:
        species_score = 0.95 if "Mastomys_natalensis" in species else 0.15
        confidence_score = max(confidence) if confidence else 0.5
        clinical_score = 0.5
        environmental_score = 0.5
        
        weighted_score = (
            self.risk_weights["species_risk"] * species_score +
            self.risk_weights["detection_confidence"] * confidence_score +
            self.risk_weights["clinical_proximity"] * clinical_score +
            self.risk_weights["environmental_factors"] * environmental_score
        )
        return max(0.0, min(1.0, weighted_score * odu_factor))
    
    def _calculate_consciousness(self, detection_data: Dict, clinical: Optional[Dict], risk_score: float) -> Dict:
        completeness = 0.8
        pattern_strength = 0.5
        risk_amplification = risk_score * 1.2
        awareness_level = max(0.0, min(1.0, (completeness + pattern_strength + risk_amplification) / 3.0))
        
        return {
            "awareness_level": round(awareness_level, 3),
            "consciousness_state": "ALERT" if awareness_level > 0.6 else "AWARE",
            "consciousness_active": awareness_level > self.consciousness_threshold
        }
    
    def _build_reasoning_chain(self, species: List[str], odu: str, risk_score: float, clinical: Optional[Dict]) -> List[str]:
        chain = [f"Ifá divination reveals Odu: {odu}", f"N-AHP consciousness risk assessment: {risk_score:.3f}"]
        if "Mastomys_natalensis" in species: chain.append("M. natalensis detected - primary Lassa reservoir species")
        chain.append("Ubuntu guidance: Community health affects individual wellbeing")
        return chain
    
    def _generate_ubuntu_recommendations(self, odu: str, risk_score: float, clinical: Optional[Dict], environmental: Optional[Dict]) -> List[str]:
        recommendations = [ODU_HEALTH_WISDOM.get(odu, {}).get('health_action', "Monitor situation")]
        if risk_score > 0.75: recommendations.append("Immediate community health education campaign")
        recommendations.append("Engage community leaders in health protection planning")
        return list(set(recommendations))
    
    def _determine_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.80: return RiskLevel.CRITICAL.value
        elif risk_score >= 0.65: return RiskLevel.HIGH.value
        elif risk_score >= 0.40: return RiskLevel.MEDIUM.value
        else: return RiskLevel.LOW.value
    
    def _identify_risk_factors(self, species: List[str], clinical: Optional[Dict]) -> List[str]:
        factors = []
        if "Mastomys_natalensis" in species: factors.append("Primary Lassa reservoir species detected")
        return factors
    
    def _check_endemic_zone(self, location: Dict) -> bool:
        return 4.0 <= location.get('latitude', 0) <= 13.0
    
    def _assess_seasonal_risk(self, environmental: Optional[Dict]) -> str:
        return "elevated" # Default
    
    def _emergency_fallback_analysis(self, detection_data: Dict) -> Dict:
        return {"status": "fallback_mode", "risk_assessment": {"level": "high", "score": 0.8}}

consciousness_engine = RemostarEngine()

def analyze_detection_consciousness(detection_data: Dict, clinical_context: Optional[Dict] = None, environmental_context: Optional[Dict] = None) -> Dict:
    return consciousness_engine.analyze_detection(detection_data, clinical_context, environmental_context)
