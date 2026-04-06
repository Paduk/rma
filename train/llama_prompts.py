SFT_RMA_TRAIN_PHI4 = """<|system|>Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|end|><|user|>{data}<|end|><|assistant|>{answer}<|end|><|endoftext|>"""
SFT_RMA_INFERENCE_PHI4 = """<|system|>Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|end|><|user|>{data}<|end|><|assistant|>"""

SFT_RMA_TRAIN_QWEN3 = """<|im_start|>system
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|im_end|>
<|im_start|>user
{data}<|im_end|><|im_start|>assistant{answer}<|im_end|>"""

SFT_RMA_INFERENCE_QWEN3 = """<|im_start|>system
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|im_end|>
<|im_start|>user
{data}<|im_end|><|im_start|>assistant"""

SFT_HISTORY_TRAIN_QWEN25 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}
<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

SFT_HISTORY_INFERENCE_QWEN25 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
"""

SFT_REWRITE_TRAIN_QWEN25 = """<|im_start|>system
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}
<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

SFT_REWRITE_INFERENCE_QWEN25 = """<|im_start|>system
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
"""

SFT_HISTORY_INFERENCE_QWEN3 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

SFT_REWRITE_INFERENCE_QWEN3 = """<|im_start|>system
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

ZERO_HISTORY_INFERENCE_QWEN25 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else. 
Required single-line format:
{{"plan":"<ToolName>","arguments":{{<Parameter1>:<Value1>,<Parameter2>:<Value2>}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
"""

ZERO_REWRITE_INFERENCE_QWEN25 = """<|im_start|>system
Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else.
Required single-line format:
{{"plan":"<ToolName>","arguments":{{<Parameter1>:<Value1>,<Parameter2>:<Value2>}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
"""

ZERO_HISTORY_INFERENCE_QWEN3 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else. 
Required single-line format:
{{"plan":"<ToolName>","arguments":{{<Parameter1>:<Value1>,<Parameter2>:<Value2>}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

ZERO_REWRITE_INFERENCE_QWEN3 = """<|im_start|>system
Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else.
Required single-line format:
{{"plan":"<ToolName>","arguments":{{<Parameter1>:<Value1>,<Parameter2>:<Value2>}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

ZERO_HISTORY_INFERENCE_PHI4 = """<|system|> You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else. 
Required single-line format:
{{"plan":"<ToolName>","arguments":{{<Parameter1>:<Value1>,<Parameter2>:<Value2>}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
<|tool|>{tools}<|/tool|><|end|><|user|>Conversation History: {conversation_history}\nUser Query: {data}<|end|><|assistant|>"""

ZERO_REWRITE_INFERENCE_PHI4 = """<|system|> Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else.
Required single-line format:
{{"plan":"<ToolName>","arguments":{{<Parameter1>:<Value1>,<Parameter2>:<Value2>}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
<|tool|>{tools}<|/tool|><|end|><|user|>User Query: {data}<|end|><|assistant|>"""

SFT_REWRITE_INFERENCE_PHI4 = """<|system|>Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}<|/tool|><|end|><|user|>User Query: {data}<|end|><|assistant|>"""

SFT_HISTORY_INFERENCE_PHI4 = """<|system|>You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}<|/tool|><|end|><|user|>Conversation History: {conversation_history}\nUser Query: {data}<|end|><|assistant|>"""

SFT_HISTORY_TRAIN_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
Conversation History: {conversation_history}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
"""

ZERO_HISTORY_INFERENCE_LLAMA = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else. 
Required single-line format:
{{"plan":"<ToolName>","arguments":{{"<Parameter1>":"<Value1>","<Parameter2>":"<Value2>"}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
Conversation History: {conversation_history}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

ZERO_REWRITE_INFERENCE_LLAMA = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> Given a user query and a list of available tools, pick the single most appropriate tool and provide its parameters. Return exactly **one** inline JSON object (no line breaks) and nothing else.
Required single-line format:
{{"plan":"<ToolName>","arguments":{{"<Parameter1>":"<Value1>","<Parameter2>":"<Value2>"}}}}

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.

Rules:
• "plan" must be a tool name from the list;
• If the tool needs no parameters, set "arguments" to {{}}.
• Output absolutely no text outside the JSON object.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SFT_HISTORY_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
Conversation History: {conversation_history}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SFT_REWRITE_TRAIN_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
"""

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

SFT_RMA_TRAIN_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history.
Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
"""

SFT_RMA_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history.
Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SFT_REWRITE_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ZERO_HISTORY_INFERENCE_LLAMA = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
# Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

# 1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
# 2. If the query does not match any of the provided tools, set the plan to "None".
# 3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
# 4. The generated data must follow the format of the example provided for that tool.
# 5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
# 6. You may refer to the conversation_history to generate an appropriate response.
# 7. output the response strictly as pure JSON in the exact format provided below. Do not include any additional text
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# Below is the information on the tools I provided:
# Tools: {tools}

# Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
# format: {re_format}
# The result should be in JSON format, following the same structure as the format above.    
# Do not create any parameter values that cannot be inferred from the user’s query.

# conversation_history: {conversation_history}
# query: {data}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

# ZERO_REWRITE_INFERENCE_LLAMA = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
# Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

# 1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
# 2. If the query does not match any of the provided tools, set the plan to "None".
# 3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
# 4. The generated data must follow the format of the example provided for that tool.
# 5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
# 6. output the response strictly as pure JSON in the exact format provided below. Do not include any additional text
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# Below is the information on the tools I provided:
# Tools: {tools}

# Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
# format: {re_format}
# The result should be in JSON format, following the same structure as the format above.    
# Do not create any parameter values that cannot be inferred from the user’s query.

# query: {data}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """