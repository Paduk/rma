SFT_RMA_TRAIN_GEMMA = """
<bos><start_of_turn>user
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history.
Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.

{data}
<end_of_turn>
<start_of_turn>model
{answer} <end_of_turn>
"""

SFT_RMA_INFERENCE_GEMMA = """
<bos><start_of_turn>user
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history.
Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.

{data}
<end_of_turn>
<start_of_turn>model
"""


SFT_REWRITE_TRAIN_GEMMA = """
<bos><start_of_turn>user
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
Tools: {tools}

Query:
{data}
<end_of_turn>
<start_of_turn>model
{answer} <end_of_turn>
"""

SFT_REWRITE_INFERENCE_GEMMA = """
<bos><start_of_turn>user
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
Tools: {tools}

Query:
{data}
<end_of_turn>
<start_of_turn>model
"""

SFT_HISTORY_TRAIN_GEMMA = """
<bos><start_of_turn>user
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Use information from the conversation history when relevant. 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.

Tools: {tools}

conversation_history: {conversation_history}
Query: {data}
<end_of_turn>
<start_of_turn>model
{answer} <end_of_turn>
"""

SFT_HISTORY_INFERENCE_GEMMA = """
<bos><start_of_turn>user
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Use information from the conversation history when relevant. 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.

Tools: {tools}

conversation_history: {conversation_history}
Query: {data}
<end_of_turn>
<start_of_turn>model
"""

ZERO_REWRITE_INFERENCE_GEMMA = """
<bos><start_of_turn>user
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
6. output the response strictly as pure JSON in the exact format provided below. Do not include any additional text

Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

Query: {data}
<end_of_turn>
<start_of_turn>model
"""

ZERO_HISTORY_INFERENCE_GEMMA = """
<bos><start_of_turn>user
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
6. You may refer to the conversation_history to generate an appropriate response.
7. output the response strictly as pure JSON in the exact format provided below. Do not include any additional text

Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

conversation_history: {conversation_history}
Query: {data}
<end_of_turn>
<start_of_turn>model
"""