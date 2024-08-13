from openai import OpenAI
from lib.CyberMetricEvaluator import CyberMetricEvaluator
from lib.utils.arguments import cmd_args

class OpenAICyberMetricEvaluator(CyberMetricEvaluator):
    def __init__(self, api_key, model_name, test_dataset_path, save_evaluation, verbose):
        self.client = OpenAI(api_key=api_key)
        super().__init__(
          model_name=model_name, 
          query_client=self.query_client, 
          get_client_response=self.get_client_response, 
          test_dataset_path=test_dataset_path, 
          save_evaluation=save_evaluation,
          verbose=verbose
        )
        
    def query_client(self, model, messages):
      return self.client.chat.completions.create(model, messages)
  
    def get_client_response(self, client_output):
      if client_output and client_output.choices and client_output.choices[0]:
        return client_output.choices[0].message.content
      else:
        return None

# Example usage:
if __name__ == "__main__":
    evaluator = OpenAICyberMetricEvaluator(
      api_key = '<YOUR-APKI-KEY-HERE>', # for test purposes only, use 'dotenv' to abstract your keys in production
      model_name = cmd_args().model if cmd_args().model else 'gpt-3.5-turbo-0125', 
      test_dataset_path = f"{cmd_args().test}.json" if cmd_args().test else 'CyberMetric-500-v1.json',
      save_evaluation = True, # Optional: Save results to "evaluations" folder
      verbose = False # Optional: Display incorrect answers in terminal
    )
    evaluator.run_evaluation()