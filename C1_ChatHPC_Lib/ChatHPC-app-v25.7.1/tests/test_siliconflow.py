import unittest
from unittest.mock import patch, MagicMock
import time
from requests.exceptions import Timeout, ConnectionError, HTTPError
from chathpc.app.siliconflow_interface import ChatHPCSiliconFlow
from chathpc.app.app import AppConfig


class TestSiliconFlowRetry(unittest.TestCase):
    def setUp(self):
        # Create a minimal AppConfig for testing
        self.config = AppConfig(
            data_file="tests/files/config.json",
            base_model_path="basemodels",
            finetuned_model_path="output",
            merged_model_path="output",
            max_response_tokens=600,
            max_training_tokens=600,
            prompt_history_file="~/.chathpc_history",
            prompt_template_file="prompt_template.txt"
        )
        
        # Mock environment variables
        import os
        os.environ["SILICONFLOW_API_KEY"] = "test-api-key"
        os.environ["SILICONFLOW_API_URL"] = "https://api.siliconflow.cn/v1/chat/completions"
        
        self.siliconflow_client = ChatHPCSiliconFlow(self.config)
    
    def test_normal_execution_no_retry(self):
        """Test normal execution without retries"""
        with patch('requests.post') as mock_post:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Test response"
                    }
                }]
            }
            mock_post.return_value = mock_response
            
            # Call the function
            result = self.siliconflow_client.siliconflow_chat_evaluate(
                "Pro/deepseek-ai/DeepSeek-V3.2",
                prompt="What is 2+2?",
                context="You are a math tutor"
            )
            
            # Verify the result
            self.assertEqual(result, "Test response")
            # Verify requests.post was called once
            self.assertEqual(mock_post.call_count, 1)
    
    def test_network_error_with_retry_success(self):
        """Test network error with retry that eventually succeeds"""
        with patch('requests.post') as mock_post:
            # First two calls raise Timeout, third call succeeds
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Test response"
                    }
                }]
            }
            
            # Make the first two calls raise Timeout, third call succeeds
            mock_post.side_effect = [
                Timeout("Connection timed out"),
                ConnectionError("Connection error"),
                mock_response
            ]
            
            # Call the function with max_retries=3
            result = self.siliconflow_client.siliconflow_chat_evaluate(
                "Pro/deepseek-ai/DeepSeek-V3.2",
                max_retries=3,
                prompt="What is 2+2?",
                context="You are a math tutor"
            )
            
            # Verify the result
            self.assertEqual(result, "Test response")
            # Verify requests.post was called three times
            self.assertEqual(mock_post.call_count, 3)
    
    def test_max_retries_exceeded(self):
        """Test when max retries are exceeded"""
        with patch('requests.post') as mock_post:
            # All calls raise Timeout
            mock_post.side_effect = Timeout("Connection timed out")
            
            # Call the function with max_retries=2
            result = self.siliconflow_client.siliconflow_chat_evaluate(
                "Pro/deepseek-ai/DeepSeek-V3.2",
                max_retries=2,
                prompt="What is 2+2?",
                context="You are a math tutor"
            )
            
            # Verify the result is None
            self.assertIsNone(result)
            # Verify requests.post was called max_retries + 1 times
            self.assertEqual(mock_post.call_count, 3)  # 2 retries + 1 initial attempt
    
    def test_non_network_error_no_retry(self):
        """Test non-network error that should not be retried"""
        with patch('requests.post') as mock_post:
            # Raise HTTPError (401 Unauthorized)
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = HTTPError("401 Client Error: Unauthorized for url")
            mock_post.return_value = mock_response
            
            # Call the function
            result = self.siliconflow_client.siliconflow_chat_evaluate(
                "Pro/deepseek-ai/DeepSeek-V3.2",
                prompt="What is 2+2?",
                context="You are a math tutor"
            )
            
            # Verify the result is None
            self.assertIsNone(result)
            # Verify requests.post was called once (no retries)
            self.assertEqual(mock_post.call_count, 1)
    
    def test_exponential_backoff(self):
        """Test exponential backoff strategy"""
        with patch('requests.post') as mock_post, patch('time.sleep') as mock_sleep:
            # Set up mock to raise Timeout twice, then succeed
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Test response"
                    }
                }]
            }
            mock_post.side_effect = [
                Timeout("Connection timed out"),
                Timeout("Connection timed out"),
                mock_response
            ]
            
            # Call the function
            result = self.siliconflow_client.siliconflow_chat_evaluate(
                "Pro/deepseek-ai/DeepSeek-V3.2",
                max_retries=3,
                prompt="What is 2+2?",
                context="You are a math tutor"
            )
            
            # Verify the result
            self.assertEqual(result, "Test response")
            # Verify requests.post was called three times
            self.assertEqual(mock_post.call_count, 3)
            # Verify time.sleep was called twice with exponential backoff
            self.assertEqual(mock_sleep.call_count, 2)
            # Check that sleep times are increasing (exponential backoff)
            sleep_calls = mock_sleep.call_args_list
            self.assertGreater(sleep_calls[1][0][0], sleep_calls[0][0])


if __name__ == "__main__":
    unittest.main()