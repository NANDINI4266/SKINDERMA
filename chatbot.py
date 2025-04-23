
import re
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HealthChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.knowledge_base = {
            'acne': {
                'symptoms': '''Detailed symptoms of acne include:
                - Whiteheads (closed plugged pores)
                - Blackheads (open plugged pores)
                - Papules (small red, tender bumps)
                - Pustules (pimples with pus at their tips)
                - Nodules (large, painful lumps beneath the skin)
                - Cystic lesions (painful, pus-filled lumps)
                - Oily skin texture
                - Scarring
                - Skin discoloration
                ''',
                'treatment': '''Comprehensive treatment options include:
                - Topical treatments:
                  • Benzoyl peroxide (kills bacteria)
                  • Retinoids (unclogs pores)
                  • Salicylic acid (reduces inflammation)
                  • Azelaic acid (kills bacteria)
                - Oral medications:
                  • Antibiotics (tetracyclines)
                  • Isotretinoin (severe cases)
                  • Birth control pills (hormonal acne)
                - Natural remedies:
                  • Tea tree oil
                  • Green tea extract
                  • Aloe vera
                ''',
                'prevention': '''Prevention tips:
                - Wash face twice daily
                - Use non-comedogenic products
                - Avoid touching face
                - Clean phone screen regularly
                - Change pillowcase weekly
                - Maintain healthy diet
                - Stay hydrated
                - Manage stress
                '''
            },
            'hyperpigmentation': {
                'symptoms': '''Hyperpigmentation presents as:
                - Dark spots or patches
                - Post-inflammatory marks
                - Melasma (larger patches)
                - Sun spots/age spots
                - Uneven skin tone
                - Freckles
                - Darkening around healed injuries
                ''',
                'treatment': '''Treatment approaches include:
                - Topical treatments:
                  • Hydroquinone (skin lightening)
                  • Kojic acid
                  • Vitamin C serums
                  • Retinoids
                  • Alpha arbutin
                - Professional treatments:
                  • Chemical peels
                  • Laser therapy
                  • Microdermabrasion
                  • IPL (Intense Pulsed Light)
                - Combination therapy
                ''',
                'prevention': '''Prevention strategies:
                - Use broad-spectrum sunscreen
                - Avoid sun exposure
                - Wear protective clothing
                - Don't pick at skin
                - Treat inflammation quickly
                - Use gentle skincare products
                '''
            },
            'vitiligo': {
                'symptoms': '''Vitiligo characteristics include:
                - White patches on skin
                - Loss of color in the mouth
                - Premature whitening of hair
                - Loss of color in eyebrows/eyelashes
                - Symmetrical pattern development
                - Progressive color loss
                - Can affect any body area
                ''',
                'treatment': '''Treatment options include:
                - Medical treatments:
                  • Topical corticosteroids
                  • Calcineurin inhibitors
                  • Phototherapy (UVB)
                  • PUVA therapy
                - Surgical options:
                  • Skin grafting
                  • Melanocyte transplants
                  • Micropigmentation
                - Combination approaches
                ''',
                'management': '''Management strategies:
                - Regular dermatologist visits
                - Sun protection
                - Emotional support
                - Stress management
                - Healthy lifestyle
                - Monitor progression
                '''
            }
        }

        self.response_templates = {
            'greeting': self._get_time_based_greeting(),
            'emergency': '🚨 MEDICAL EMERGENCY: Please seek immediate medical attention!',
            'consult': '👨‍⚕️ Please consult a healthcare professional for proper diagnosis and treatment.',
            'disclaimer': '⚠️ This information is for educational purposes only and not a substitute for professional medical advice.',
            'followup': ['Would you like to know more about treatments?', 
                        'Should I explain prevention methods?',
                        'Would you like to see related conditions?']
        }

        # Prepare vectorizer with all possible responses
        self.all_responses = []
        for condition in self.knowledge_base.values():
            for info in condition.values():
                self.all_responses.append(info)
        self.vectorizer.fit(self.all_responses)

    def _get_time_based_greeting(self):
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning! 🌅"
        elif 12 <= hour < 17:
            return "Good afternoon! ☀️"
        elif 17 <= hour < 22:
            return "Good evening! 🌆"
        else:
            return "Hello! Working late? 🌙"

    def _find_best_response(self, question):
        """Use TF-IDF and cosine similarity to find best matching response"""
        question_vector = self.vectorizer.transform([question])
        all_responses_vector = self.vectorizer.transform(self.all_responses)
        similarities = cosine_similarity(question_vector, all_responses_vector)
        best_match_index = similarities.argmax()
        confidence = similarities[0][best_match_index]
        return self.all_responses[best_match_index], confidence

    def _format_response(self, response, confidence=None):
        """Enhanced formatting for responses"""
        confidence_str = f"\nConfidence: {confidence*100:.1f}%" if confidence else ""
        return f"""
        <div style='background-color: #e9ecef; 
                    border-left: 4px solid #2196F3; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1)'>
            {response}{confidence_str}
        </div>
        """

    def get_answer(self, question):
        question = question.lower()
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Check for greetings
        if any(greeting in question for greeting in ['hi', 'hello', 'hey']):
            return self._format_response(f"[{timestamp}] {self._get_time_based_greeting()}")

        # Check for emergencies
        emergency_keywords = ['emergency', 'severe pain', 'bleeding', 'unconscious']
        if any(keyword in question for keyword in emergency_keywords):
            return self._format_response(f"[{timestamp}] {self.response_templates['emergency']}")

        # Find best matching response using TF-IDF
        best_response, confidence = self._find_best_response(question)

        # Add follow-up suggestion if confidence is high
        if confidence > 0.3:
            followup = random.choice(self.response_templates['followup'])
            best_response += f"\n\n💡 {followup}"

        # Format and return response
        formatted_response = f"[{timestamp}]\n{best_response}\n\n{self.response_templates['disclaimer']}"
        return self._format_response(formatted_response, confidence)

    def get_suggestions(self):
        """Get contextual suggestions based on common queries"""
        return [
            "What are the symptoms of acne? 🔍",
            "How can I treat hyperpigmentation? 💊",
            "What causes vitiligo? 🤔",
            "Tell me about acne prevention 🛡️",
            "What are natural remedies for acne? 🌿",
            "How to prevent hyperpigmentation? ☀️",
            "Is vitiligo treatment effective? 💉",
            "Tell me about acne scarring 📋",
            "What triggers hyperpigmentation? ⚡",
            "How to manage vitiligo? 🔬",
            "What foods should I avoid for acne? 🍔",
            "Are there new treatments for vitiligo? 🔬",
            "How long does hyperpigmentation last? ⏳",
            "Is acne hormonal? 🧪",
            "Can stress cause skin problems? 😰"
        ]

    def get_dynamic_suggestions(self, current_query):
        """Get suggestions based on current query context"""
        condition_matches = {
            'acne': ['treatment', 'prevention', 'causes', 'food', 'hormonal', 'scars'],
            'hyperpigmentation': ['sun', 'dark spots', 'melasma', 'treatment', 'prevention'],
            'vitiligo': ['white patches', 'treatment', 'causes', 'management', 'progression']
        }

        suggestions = []
        for condition, keywords in condition_matches.items():
            if any(keyword in current_query.lower() for keyword in keywords):
                suggestions.extend([
                    f"What are the symptoms of {condition}?",
                    f"How is {condition} treated?",
                    f"What causes {condition}?",
                    f"How can I prevent {condition}?",
                    f"Are there natural remedies for {condition}?"
                ])

        return suggestions[:5] if suggestions else self.get_suggestions()[:5]
