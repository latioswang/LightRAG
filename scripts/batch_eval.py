import re
import json
import jsonlines
import pandas as pd
from openai import OpenAI
import argparse
import asyncio
from lightrag.llm import gpt_4o_complete, gpt_4o_mini_complete

def parse_evaluation_json(text):
    """Extract and parse JSON from text, looking for content between curly braces."""
    try:
        # Try to parse the text directly first
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find JSON pattern
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse valid JSON from response: {text}")
        raise ValueError(f"No JSON pattern found in response: {text}")

async def batch_eval(csv_file, column_query, column_r1, column_r2, output_file_path):
    client = OpenAI()

    # Read CSV file
    df = pd.read_csv(csv_file, header=0, skiprows=lambda x: x > 0 and x <= int(args.skip_lines))
    
    # Validate column names
    required_columns = [column_query, column_r1, column_r2]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    queries = df[column_query].tolist()
    answers1 = df[column_r1].tolist()
    answers2 = df[column_r2].tolist()

    results = []
    stats = {
        "Comprehensiveness": {"Answer 1": 0, "Answer 2": 0},
        "Diversity": {"Answer 1": 0, "Answer 2": 0},
        "Empowerment": {"Answer 1": 0, "Answer 2": 0},
        "Overall Winner": {"Answer 1": 0, "Answer 2": 0}
    }
    
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.
        Note, do not make conslusion just because one answer has more words or is more formal, choose the one that is having actual meaningful content and is more comprehensive, diverse, and empowering.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        response = await gpt_4o_complete(
            prompt=prompt,
            system_prompt=sys_prompt
        )
        
        try:
            evaluation = parse_evaluation_json(response)
            
            # Update statistics
            for criterion in ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]:
                winner = evaluation[criterion]["Winner"]
                stats[criterion][winner] += 1

        except (ValueError, KeyError) as e:
            print(f"Warning: Could not parse evaluation for comparison {i+1}: {str(e)}")
            evaluation = response  # Store raw response if parsing fails

        result = {
            "request_id": f"request-{i+1}",
            "query": query,
            "answer1": answer1,
            "answer2": answer2,
            "evaluation": evaluation
        }
        results.append(result)
        print(f"Processed comparison {i+1}/{len(queries)}")

    # Calculate and display statistics
    total = len(queries)
    summary = {
        "total_comparisons": total,
        "statistics": {}
    }
    
    print("\nEvaluation Summary:")
    for criterion in stats:
        a1_wins = stats[criterion]["Answer 1"]
        a2_wins = stats[criterion]["Answer 2"]
        a1_rate = (a1_wins / total) * 100 if total > 0 else 0
        a2_rate = (a2_wins / total) * 100 if total > 0 else 0
        
        print(f"{criterion}:")
        print(f"  {column_r1} win rate: {a1_rate:.1f}% ({a1_wins}/{total})")
        print(f"  {column_r2} win rate: {a2_rate:.1f}% ({a2_wins}/{total})")
        
        summary["statistics"][criterion] = {
            f"{column_r1}_win_rate": a1_rate,
            f"{column_r2}_win_rate": a2_rate,
            f"{column_r1}_wins": a1_wins,
            f"{column_r2}_wins": a2_wins
        }

    # Write results to output file
    with jsonlines.open(output_file_path, mode="w") as writer:
        writer.write({
            "metadata": {
                "csv_file": csv_file,
                "column_query": column_query,
                "column_r1": column_r1,
                "column_r2": column_r2,
                "summary": summary
            }
        })
        for result in results:
            writer.write(result)

    print(f"\nEvaluation results written to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate answers using GPT-4.')
    parser.add_argument('--csv_file', help='Path to the CSV file containing queries and answers', default='data/questions.csv')
    parser.add_argument('--column_query', help='Name of the column containing queries', default='Full Text')
    parser.add_argument('--column_r1', help='Name of the column containing first answers', default='GPT-4')
    parser.add_argument('--column_r2', help='Name of the column containing second answers', default='KB')
    parser.add_argument('--output', help='Path to output file', default='output.jsonl')
    parser.add_argument('--skip_lines', help='Number of lines to skip', default=11)
    
    args = parser.parse_args()
    
    asyncio.run(batch_eval(
        args.csv_file,
        args.column_query,
        args.column_r1,
        args.column_r2,
        args.output
    ))
