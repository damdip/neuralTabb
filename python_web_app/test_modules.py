import unittest
from unittest.mock import MagicMock, patch
from modules import GeminiClient, ChatHistory

class TestGeminiClient(unittest.TestCase):

    @patch('modules.genai.configure')
    @patch('modules.genai.GenerativeModel')
    def setUp(self, mock_generative_model, mock_configure):
        """Configura un client Gemini fittizio per i test."""
        self.api_key = "test_api_key"
        
        # Mock della configurazione e del modello Gemini
        self.mock_model_instance = MagicMock()
        mock_generative_model.return_value = self.mock_model_instance
        
        self.gemini_client = GeminiClient(api_key=self.api_key)

    def test_initialization(self):
        """Verifica che il client sia inizializzato correttamente."""
        self.assertEqual(self.gemini_client.api_key, self.api_key)
        self.assertIsNotNone(self.gemini_client.model)
        self.assertIsInstance(self.gemini_client.chat_history, ChatHistory)

    def test_classify_question(self):
        """Testa la classificazione di una domanda."""
        question = "quanti sono i libri nella collezione?"
        expected_classification = "analytical"
        
        # Mock della risposta del modello
        mock_response = MagicMock()
        mock_response.text = expected_classification
        self.mock_model_instance.generate_content.return_value = mock_response
        
        classification = self.gemini_client.classify_question(question)
        
        self.assertEqual(classification, expected_classification)
        self.mock_model_instance.generate_content.assert_called_once()

    def test_handle_conversational_question(self):
        """Testa la gestione di una domanda conversazionale."""
        question = "ciao, come stai?"
        expected_response = "Sto bene, grazie!"
        
        # Mock della risposta del modello
        mock_chat_session = MagicMock()
        mock_chat_session.send_message.return_value.text = expected_response
        self.mock_model_instance.start_chat.return_value = mock_chat_session
        
        response = self.gemini_client.handle_conversational_question(question)
        
        self.assertEqual(response, expected_response)
        self.assertEqual(len(self.gemini_client.chat_history.get_history()), 2) # Utente e modello

if __name__ == '__main__':
    unittest.main()

