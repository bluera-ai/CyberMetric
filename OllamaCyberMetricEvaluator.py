import ollama
from lib.CyberMetricEvaluator import CyberMetricEvaluator
from lib.utils.arguments import cmd_args

class OllamaCyberMetricEvaluator(CyberMetricEvaluator):
    def __init__(self, model_name, test_dataset_path, save_evaluation, verbose):
        super().__init__(
          model_name=model_name, 
          query_client=self.query_client, 
          get_client_response=self.get_client_response, 
          test_dataset_path=test_dataset_path, 
          save_evaluation=save_evaluation,
          verbose=verbose
        )
      
    @staticmethod  
    def query_client(model, messages):
      return ollama.chat(model, messages)
  
    def get_client_response(self, client_output):
      if client_output and client_output['message'] and client_output['message']['content']:
        return client_output['message']['content']
      else:
        return None

# Example usage:
if __name__ == "__main__":
    evaluator = OllamaCyberMetricEvaluator(
      model_name = cmd_args().model if cmd_args().model else 'llama3.1:70b', 
      test_dataset_path = f"{cmd_args().test}.json" if cmd_args().test else 'CyberMetric-80-v1.json',
      save_evaluation = True, # Optional: Save results to "evaluations" folder
      verbose = False # Optional: Display incorrect answers in terminal
    )
    evaluator.run_evaluation()
    