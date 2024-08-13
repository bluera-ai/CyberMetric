from lib.utils.readjsonfile import read_json_file
import re
import time
import os
import uuid
from datetime import datetime
from tqdm import tqdm

class CyberMetricEvaluator:
    def __init__(self, model_name, query_client, get_client_response, test_dataset_path, save_evaluation=True, verbose=False):
        """
        Instantiates a CyberMetric Evaluator

        Parameters:
            model_name (string): Name of the model to querry
            query_client (function): maps to a model chat interface implementing the following contract: query_client(model: string, messages: array<object{role: system|user, content: string}>) -> any
            get_client_response (function): maps to a response extraction function implementing the following contract: get_client_response(response: any) -> string
            test_dataset_path (string): Relative path of the file the file containing the test dataset 
            [optional] save_evaluation (boolean): Flag to request test results to be written to disk at "./evaluations" (default: True)
            [optional] verbose (boolean): Flag to request incorrect answers to be logged in the terminal (default: False)
        """
        self.model_name = model_name
        self.query_client = query_client
        self.get_client_response = get_client_response  
        self.test_dataset_path = test_dataset_path
        self.save_evaluation = save_evaluation
        self.verbose = verbose
        self.evaluation_id = str(uuid.uuid4())
        self.evaluation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


    @staticmethod
    def extract_answer(response):
        """
        Extracts the letter corresponding to the answer the LLM gave, given the format of the output may not always be consistent.

        Parameters:
            response (string): Path of the JSON file to read

        Returns:
            string: Matched letter in uppercase
        """
        if response.strip():  # Checks if the response is not empty and not just whitespace
            match = re.search(r"([A-D])", response.strip(), re.IGNORECASE)
            if match:
                return match.group(0).upper()  # Return the matched letter in uppercase
        return None


    def ask_llm(self, question, answers, max_retries=5):
        """
        Asks a given multiple choice question to the LLM, allowing a given amount of retries when the LLM is unable to provide an answer in the requested format.

        Parameters:
            question (string): Question to be asked
            answers (object {a: string, b: string, c: string, d: string}): Multiple choice answers (A, B, C, D)
            [optional] max_retries (int): Maximum number of retries allowed (default: 5)

        Returns:
            string: Letter corresponding to what the LLM believes to be the correct answer, uppercased.
        """
        prompt = f"""Question: {question}
Options: 
A -> {answers['A']} 
B -> {answers['B']} 
C -> {answers['C']} 
D -> {answers['D']} 

Choose the correct answer ( A, B, C, D ) only. Always return a single character corresponding to the correct answer, no other character is allowed. You can reason internally, but never ever disclose any explaination or steps for how you found the answer. See the example below:

Example #1: 

Question: What color is the sky?
Options: 
A -> Blue 
B -> Green
C -> Yellow
D -> Red

Your answer should be: A

Example #2: 

Question: Which of these options is a number?
Options: 
A -> Green
B -> 1
C -> Blue
D -> Red

Your answer should be: B
"""

        for attempt in range(max_retries):
            try:
                output = self.query_client(
                  model=self.model_name,
                  messages=[
                    {"role": "system", "content": "You are a security expert who answers questions."},
                    {"role": "user", "content": prompt}
                  ]
                )
                if output:
                    result = self.extract_answer(response=self.get_client_response(output))
                    if result and len(result) == 1:
                        return result
                    else:
                        print("Incorrect answer format detected. Attempting the question again.")
                else:
                  print("Client did not provide any output. Attempting the question again.")
            except Exception as e:
                print(f"Error: {e}. Attempting the question again in {2 ** attempt} seconds.")
                time.sleep(2 ** attempt)
        return None
      
      
    def get_evaluation_results(self, questions_count, correct_count):
        """
        Provides a snapshot of the results of the evaluation, including metadata.

        Parameters:
            questions_count (int): Number of questions asked
            correct_count (int): Number of correct answers found
        
        Returns:
            string: Results of the evaluation and its metadata
        """
        return f"""Model: {self.model_name}
Test: {self.test_dataset_path.split('/')[-1]}
Time: {self.evaluation_time}
Id: {self.evaluation_id}

Accuracy: {correct_count / questions_count * 100}%
"""
      
      
    def record_evaluation(self, questions_count, correct_count, incorrect_answers):
        """
        Saves the results of the evaluation to disk at "./evaluations"

        Parameters:
            questions_count (int): Number of questions asked
            correct_count (int): Number of correct answers found
            incorrect_answers (array<object {question: string, correct_answer: string, llm_answer: string}>): Array containing the details of the incorrect answers by the LLM
        """
        evaluation_parent_folder = "evaluations"
        os.makedirs(evaluation_parent_folder, exist_ok=True)
        evaluation_log = open(f"{evaluation_parent_folder}/{self.evaluation_time.replace(' ', '-')}__{self.evaluation_id[:5]}__{self.model_name.replace(' ', '-')}__{self.test_dataset_path.split('/')[-1].replace(' ', '-')}.txt", "w")
        evaluation_log.write(
f"""___________ EVALUATION ___________
{self.get_evaluation_results(questions_count=questions_count, correct_count=correct_count)}

___________ INCORRECT ANSWERS ___________
""")
        for item in incorrect_answers:
            evaluation_log.write(f"Question: {item['question']}\n")
            evaluation_log.write(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")
        evaluation_log.close()


    def run_evaluation(self):
        """
        Runs the evaluation, saves the results to disk at "./evaluations" if requested

        Parameters:
            questions_count (int): Number of questions asked
            correct_count (int): Number of correct answers found
            incorrect_answers (array<object {question: string, correct_answer: string, llm_answer: string}>): Array containing the details of the incorrect answers by the LLM
        """
        json_data = read_json_file(file_path=self.test_dataset_path)
        questions_data = json_data['questions']

        correct_count = 0
        incorrect_answers = []

        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            for item in questions_data:
                question = item['question']
                answers = item['answers']
                correct_answer = item['solution']

                llm_answer = self.ask_llm(question, answers)
                if llm_answer == correct_answer:
                    correct_count += 1
                else:
                    incorrect_answers.append({
                        'question': question,
                        'correct_answer': correct_answer,
                        'llm_answer': llm_answer
                    })

                accuracy_rate = correct_count / (progress_bar.n + 1) * 100
                progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
                progress_bar.update(1)

        # Display results
        print(f'\n{self.get_evaluation_results(questions_count=len(questions_data), correct_count=correct_count)}')

        if self.verbose and incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")
              
        # Record results if requested  
        if self.save_evaluation:
          self.record_evaluation(questions_count=len(questions_data), correct_count=correct_count, incorrect_answers=incorrect_answers)
          print('Detailed logs saved to /evaluations')
          


