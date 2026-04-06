COMPLEX_REWRITE_INFERENCE_PROMPT = """
Please generate a rewrited_query based on the provided conversation_history and query.
The goal of the rewrited_query is to rewrite the original query into a clear, explicit, and fully self-contained sentence by replacing ambiguous pronouns, demonstratives, demonstrative adjectives, or referential expressions (such as "it," "that," "there," "that xml file," "that image file," "that number," "that file uri," "that contact," "that location," "her," "him") with explicit information directly from the conversation_history.

Important Guidelines:
Rule 1: Identify all ambiguous pronouns, demonstratives, demonstrative adjectives, or referential terms in the query (including next_turn_query), and explicitly replace each of them with corresponding details from the conversation_history. Especially, replace demonstrative adjectives such as "that xml file," "that image file," "that number," "that file uri," "that contact," "that location," "her," or "him" appearing in next_turn_query explicitly using conversation_history. Try to keep the original query sentence structure whenever possible.
Rule 2: Keep exactly the same sentence structure as the original query. Only replace ambiguous references with explicit values.
Rule 3: If the query references specific values such as URIs, file paths, phone numbers, email addresses, or physical addresses, you MUST explicitly include these values in the rewrited_query. Every explicit value used in the rewrited_query must appear exactly as presented in the conversation_history or query. Do NOT alter these values in any way, and do NOT generate or include any additional information that is not explicitly provided.
Rule 4: Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. No additional context, explanation, or commentary should be added.

Here are clear and correct examples demonstrating the desired format and behavior:
examples: {examples}

Now, using only the data provided below, generate the rewrited_query. Respond strictly following the JSON format given below and include no additional text or explanation:
{{"rewrited_query": ""}}
data:
{data}
"""

COMPLEX_MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query.

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request like human that fits the context.
 2. The next_turn_query should be as concise as possible and can be written in various forms, such as commands, questions, or asking with detailed descriptions. However, explanations about the next_turn_query themselves should not be provided.
 3. The generated next_turn_query, based on the query and device_response, should comprehensively cover all possible use cases of the tool, including various parameters to verify the tool's ability to handle different types of tasks. Information for parameters with required=True must be included.
 4. The generated next_turn_query should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 5. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make. 
 6. When objects or return values are included in the 'query' and 'device_response', you should utilize them only in case that represent provided tool's parameter. but avoid explicitly mentioning their values directly. Instead, refer to them using demonstrative adjective (e.g., "that xml file", "that image file", "that number", "that file uri", "that contact", "that location", "her", "him")
 7. Use demonstrative adjective to refer to one or more of the provided tool's parameters by utilizing conversation_history, query or device_response.
 8. Each next_turn_query should have only one purpose.
 9. Do not generate same next_turn_query among the provided data.

Guidelines for writing rewrited_query:
The purpose of the rewrited_query is to convert the next_turn_query into a clear sentence.
 1. Replace ambiguous demonstrative adjective used in next_turn_query (e.g., "that xml file", "that image file", "that number", "that file uri", "that contact", "that location", "her", "him") with explicit information (query or device_response) from the previous turn. do not change sentence format against multi_turn_query
 2. explicit value in rewrite_query is take it from query, device_response, next_turn_query
 3. Keep the structure and form of the next_turn_query and sentence as they are, but replace demonstrative adjective with explicit values only when the demonstrative adjective refer to conversation_history, query, or device_response. 

Again, you need to consider important rules below.
rule 1. next_turn_query covers required parameter in description using demonstrative adjective
rule 2. If conversation_history, query, and device_response include explicit values representing the provided tool parameters, then next_turn_query should contains demonstrative adjective.
rule 3. if demonstrative adjective represent provided tool's paramter, rewrited_query is same format sentence with next_turn_query, but demonstrative adjective
rule 4. explicit value in rewrite_query is take it from query, device_response, next_turn_query 
 
[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query with an appropriate value, directly to the top-level of the JSON object, not inside any nested field. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible and it should align with the intended purpose of the tool and be something that a real user would actually say or use.

[previous_turn_datas]
{previous_turn_data}
"""

ZERO_REWRITE_PROMPT = """
Please write a rewrited_query based on the given conversation_history and query.
The purpose of the rewrited_query is to convert the query into a clearer and more explicit sentence.

Rule 1: Replace ambiguous pronouns, demonstratives, or referential expressions (e.g., "it," "that," "there") in the query with explicit information from the conversation_history.
Rule 2: The rewrited_query should have the same sentence structure as the query, but replace all pronouns, demonstratives, and referential terms with explicit information.
Rule 3: Any explicit values used in the rewrited_query must be taken directly from the conversation_history.
Rule 4: Do not add any new information or values that are not present in the conversation_history or query.

format: {re_format}

now, please write a rewrited_query based on the given conversation_history and query.
Generate only json format data refer to the format above.

conversation_history: {conversation_history}
query: {query}
"""

ZERO_SHOT_SMALL_REWRITE_PROMPT = """
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.

Respond ONLY in JSON format without any explanation:
{{"rewrited_query": "<rewritten query>"}}
data: {data}
"""

FEW_SHOTS_SMALL_REWRITE_PROMPT = """
Rewrite the query clearly by replacing ambiguous pronouns emonstrative with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.
examples: {examples}

Respond ONLY in JSON format without any explanation:
{{"rewrited_query": "<rewritten query>"}}
data: {data}
"""

FEW_SHOTS_SIMPLER_REWRITE_PROMPT = """
Please generate a rewrited_query based on the provided conversation_history and query.

The goal of the rewrited_query is to rewrite the original query into a clear, explicit, and fully self-contained sentence by replacing ambiguous pronouns, demonstratives, or referential expressions (such as "it," "that," "there") with explicit information directly from the conversation_history.

Important Guidelines:
Rule 1. Rewrite the query clearly into a fully self-contained sentence, explicitly replacing ambiguous references (like 'it' or 'that') with details directly from conversation_history.
Rule 2. Keep exactly the same sentence structure.
Rule 3. Include values (e.g., URIs, phone numbers, addresses) exactly as given.
Rule 4. Do NOT add any information not explicitly provided

Here are clear and correct examples demonstrating the desired format and behavior:
examples: {examples}

Now, using only the data provided below, generate the rewrited_query. Respond strictly following the JSON format given below and include no additional text or explanation:
{{"rewrited_query": ""}}
data:
{data}
"""

FEW_SHOTS_REWRITE_PROMPT = """
Please generate a rewrited_query based on the provided conversation_history and query.

The goal of the rewrited_query is to rewrite the original query into a clear, explicit, and fully self-contained sentence by replacing ambiguous pronouns, demonstratives, or referential expressions (such as "it," "that," "there") with explicit information directly from the conversation_history.

Important Guidelines:
Rule 1: Identify all ambiguous pronouns, demonstratives, or referential terms in the query and replace each of them explicitly with corresponding details from the conversation_history.
Rule 2: Keep exactly the same sentence structure as the original query. Only replace ambiguous references with explicit values. However, exclude any natural fillers, hesitations, or conversational asides when rewriting the sentence.
Rule 3: If the query references specific values such as URIs, file paths, phone numbers, email addresses, or physical addresses, you MUST explicitly include these values in the rewrited_query. Every explicit value used in the rewrited_query must appear exactly as presented in the conversation_history or query. Do NOT alter these values in any way, and do NOT generate or include any additional information that is not explicitly provided.
Rule 4: Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. No additional context, explanation, or commentary should be added.

Here are clear and correct examples demonstrating the desired format and behavior:
examples: {examples}

Now, using only the data provided below, generate the rewrited_query. Respond strictly following the JSON format given below and include no additional text or explanation:
{{"rewrited_query": ""}}
data:
{data}
"""

ZERO_SHOT_INFERENCE_GENERATION_PROMPT = """
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.

Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

Query: {data}
"""

ZERO_SHOT_HISTORY_INFERENCE_GENERATION_PROMPT = """
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
6. You may refer to the conversation_history to generate an appropriate response.

Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

conversation_history: {conversation_history}
Query: {data}
"""

FEW_SHOT_INFERENCE_GENERATION_PROMPT = """
Please help identify the appropriate plan for a given query and assist in generating its parameters.  
I will provide descriptions and example data for several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query explicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.

Below are examples of tool descriptions and query-response pairs that I’ve provided:  
Tool-example pairs: {tools-examples}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key.  
The result should be in JSON format, following the same structure as the examples above.  
Do not create any parameter values that cannot be inferred from the user’s query.
{data}
"""

RESPONSE_GENERATION_PROMPT = """
After successfully executing a plan based on the user's query, this will help generate natural and accurate device_responses for the mobile AI assistant. You will receive detailed tool descriptions (plans), results from plan execution, previous conversation history, and the user's queries. Your task is to explicitly create clear, contextually appropriate device_responses based on these inputs.

Carefully follow the instructions below:
 1. Diversity and Successful device_response (Brief): Utilize diverse vocabulary and sentence structures to maintain naturalness with varying tones and attitudes, yet keep concise.
 2. Contextual Accuracy: Fully understand the provided tool description and user query to generate contextually accurate device_responses. 
 3. When generating a device_response for tools like VIEW_SENT_SOMETHINGS that check past records, ensure the reliability of the data by referencing the conversation_history that contains past device_response records.
 4. Please refer to the 'returns' section in the tool description for the format of the content to be written in device_response. Based on the 'returns' section, generate a variety of examples in a similar format. If conversation_history exists, you may also need to refer to previous responses.\
 5. If the return value is not explicitly stated in the user query but needs to be generated creatively, it should be produced in a diverse and non-repetitive manner.
 6. In addition, make sure to explicitly generate device_responses that clearly reflect the values returned by the executed plan. If the returned values include specific details such as URIs or phone numbers, state them clearly and accurately. 
 
Below is the provided tool (plan) including the plan and execution results:
Tool: {tool}

When generating a device_response based on the tool, each conversation_history, and query, imagine yourself as a Mobile Device. Also, make sure to vary the tone and phrasing for each generated response.
Based on the provided query-device_response sets below, clearly and appropriately add a 'device_response' field.
Do not modify any of the values provided in the query-device_response sets. Please add a new key "device_response" with an appropriate value, directly to the top-level of the JSON object, not inside any nested field
{dataset}
"""

MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request that fits the context.
 2. The queries should be as simple as possible. also a query can be a command, a question, or a request with detailed descriptions, etc. never explaining about it.
 3. The generated queries should comprehensively cover all possible uses of the tool, ensuring the coverage of various parameters, thus verifying the tool's ability to handle various forms of work.
 4. The generated queries should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 5. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make.
 6. When objects or return values are included in the 'query' and 'device_response', you must utilize them, but avoid explicitly mentioning their values directly. Instead, refer to them using pronouns such as "it," "that," "there," "her," or "him." in case that, Every next_turn_query must use pronouns.
 7. If pronouns in next_turn_query do not indicate the required parameter, next_turn_query must be generated as a sentence that includes the required parameter.

Guidelines for writing rewrited_query:
 1. Replace ambiguous pronouns used in next_turn_query (e.g., "it," "that," "there") with explicit information (query or device_response) from the previous turn.
 2. If the user's request is already clear and specific, rewrite_query should be same as next_turn_query
 3. Also If next_turn_query has not pronouns, rewrite_query should be same as next_turn_query
 4. If the next_turn_query contains pronouns and device_response contains specific values, the rewrited_query must explicitly use those specific values.
 5. The rewrited_query should be a clear and easily understandable sentence.

[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query fields to the all JSON datas below. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible.

[previous_turn_datas]
{previous_turn_data}
"""

SIMPLE_MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query.

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request like human that fits the context.
 2. The next_turn_query should be as concise as possible and can be written in various forms, such as commands, questions, or asking with detailed descriptions. However, explanations about the next_turn_query themselves should not be provided.
 3. The generated next_turn_query, based on the query and device_response, should comprehensively cover all possible use cases of the tool, including various parameters to verify the tool's ability to handle different types of tasks. Information for parameters with required=True must be included.
 4. The generated next_turn_query should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 5. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make. 
 6. When objects or return values are included in the 'query' and 'device_response', you should utilize them only in case that represent provided tool's parameter. but avoid explicitly mentioning their values directly. Instead, refer to them using pronouns such as "it," "that," "there," "her," or "him.". 
 7. Use pronouns to refer to one or more of the provided tool's parameters by utilizing conversation_history, query or device_response.
 8. Each next_turn_query should have only one purpose.
 9. Do not generate same next_turn_query among the provided data.
 
Guidelines for writing rewrited_query:
The purpose of the rewrited_query is to convert the next_turn_query into a clear sentence.
 1. Replace ambiguous pronouns used in next_turn_query (e.g., "it," "that," "there") with explicit information (query or device_response) from the previous turn. do not change sentence format against multi_turn_query
 2. explicit value in rewrite_query is take it from query, device_response, next_turn_query
 3. Keep the structure and form of the next_turn_query and sentence as they are, but replace pronouns with explicit values only when the pronouns refer to conversation_history, query, or device_response. 

Again, you need to consider important rules below.
rule 1. next_turn_query covers required parameter in description using pronouns
rule 2. If conversation_history, query, and device_response include explicit values representing the provided tool parameters, then next_turn_query should contains pronouns.
rule 3. if pronouns represent provided tool's paramter, rewrited_query is same format sentence with next_turn_query, but pronouns
rule 4. explicit value in rewrite_query is take it from query, device_response, next_turn_query 
 
[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query with an appropriate value, directly to the top-level of the JSON object, not inside any nested field. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible and it should align with the intended purpose of the tool and be something that a real user would actually say or use.

[previous_turn_datas]
{previous_turn_data}
"""

DIFFICULTY_MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query.

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request like human that fits the context.
 2. The next_turn_query should be as complicated as possible and can be written in various forms, such as commands, questions, or asking with detailed descriptions. However, explanations about the next_turn_query themselves should not be provided. 
 3. It should sound as if a real person is talking to a device—include natural fillers, hesitations, or even unnecessary remarks to make it feel more human.
 4. The generated next_turn_query, based on the query and device_response, should comprehensively cover all possible use cases of the tool, including various parameters to verify the tool's ability to handle different types of tasks. Information for parameters with required=True must be included.
 5. The generated next_turn_query should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 6. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make. 
 7. When objects or return values are included in the 'query' and 'device_response', you should utilize them only in case that represent provided tool's parameter. but avoid explicitly mentioning their values directly. Instead, refer to them using pronouns such as "it," "that," "there," "her," or "him.". 
 8. Use pronouns to refer to one or more of the provided tool's parameters by utilizing conversation_history, query or device_response.
 9. Each next_turn_query should have only one purpose.
 10. Do not generate same next_turn_query among the provided data.
 
Guidelines for writing rewrited_query:
The purpose of the rewrited_query is to convert the next_turn_query into a clear sentence.
 1. Replace ambiguous pronouns used in next_turn_query (e.g., "it," "that," "there") with explicit information (query or device_response) from the previous turn. do not change sentence format against multi_turn_query
 2. explicit value in rewrite_query is take it from query, device_response, next_turn_query
 3. Keep the structure and form of the next_turn_query and sentence as they are, but replace pronouns with explicit values only when the pronouns refer to conversation_history, query, or device_response. 

Again, you need to consider important rules below.
rule 1. next_turn_query covers required parameter in description using pronouns
rule 2. If conversation_history, query, and device_response include explicit values representing the provided tool parameters, then next_turn_query should contains pronouns.
rule 3. if pronouns represent provided tool's paramter, rewrited_query is same format sentence with next_turn_query, but pronouns
rule 4. explicit value in rewrite_query is take it from query, device_response, next_turn_query 
 
[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query with an appropriate value, directly to the top-level of the JSON object, not inside any nested field. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible and it should align with the intended purpose of the tool and be something that a real user would actually say or use.

[previous_turn_datas]
{previous_turn_data}
"""

REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT = """
Please help me generate a plan-calling dataset. I will provide a tool description and some example data. 
Based on this tool, you need to generate answers for the provided queries. In other words, you need to provide answers that call the tool to resolve the user's query. Here are my requirements:

1. For the generated queries, you must provide answers using the tool. That is, specify the tool used and the values for each parameter.
2. When providing parameters, if a parameter has required=False, it is not necessary to provide its value. However, if the query includes that parameter, you must provide it.
3. The generated data must be presented in the format given in the example above.
4. The parameter values generated with plan calls must be values that can be inferred from the user's query. You must not fabricate parameters that cannot be obtained from the user's request.
5. If a query does not include a required=True parameter or does not provide enough information, set the answer to 'N/A'."

Below are the tool I provided and some examples of query-answer pairs:
Tool: {tool}
Examples: {examples}

Now, please help me generate answer pairs for the queries below. The generated data should include both a query and an answer key.
The results must be generated in JSON format like the example above.
Do not fabricate parameters for the tool; only use parameters that can be inferred from the user's request.
{data}
"""

# debug prompt
REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT = """
Please help me generate a plan-calling dataset. I will provide a tool description and some example data. 
Based on this tool, you need to generate answers for the provided queries. In other words, you need to provide answers that call the tool to resolve the user's query. Here are my requirements:

1. For the generated queries, you must provide answers using the tool. That is, specify the tool used and the values for each parameter.
2. When providing parameters, if a parameter has required=False, it is not necessary to provide its value. However, if the query includes that parameter, you must provide it.
3. The generated data must be presented in the format given in the example above.
4. The parameter values generated with plan calls must be values that can be inferred from the user's query. You must not fabricate parameters that cannot be obtained from the user's request.
5. If a query does not include a required=True parameter, set the answer.plan value as 'N/A' and answer.arguments.reason with reasons"

Below are the tool I provided and some examples of query-answer pairs:
Tool: {tool}
Examples: {examples}

Now, please help me generate answer pairs for the queries below. The generated data should include both a query and an answer key.
The results must be generated in JSON format like the example above.
Do not fabricate parameters for the tool; only use parameters that can be inferred from the user's request.
{data}
"""

REWRITE_FILTER_PROMPT = """
If the explicit values in the rewrited_query can be inferred from conv_history and next_turn_query, return True; otherwise, return False.

belows are two example
conversation_history: Can you modify Bob's contact details? His contact URI is content://contacts/people/bob and I'd like to update his phone number to 123-4567. -> No problem, I will update Bob's phone number to 123-4567 using the provided URI content://contacts/people/bob.
next_turn_query: Send him a message saying 'Your phone details have been updated, Bob!'
rewrited_query: Send content://contacts/people/bob a message saying 'Your phone details have been updated, Bob!'
response: True
examples 1. This is True. content://contacts/people/bob in rewrited_query already used in conversation_history.


conversation_history: I would like to access the details of contact content://contacts/people/1023. Can you open its profile? -> Opening contact profile at content://contacts/people/1023.
next_turn_query: Call this contact
rewrited_query: Call 010-1023-0000
response: False
examples 2. This is False. '010-1023-0000' in rewrited_query can not infered from conversation_history and next_turn_query

If the explicit values in the rewrited_query can be inferred from conv_history and next_turn_query, return True; otherwise, return False.
conversation_history: {conversation_history}
next_turn_query: {next_turn_query}
rewrited_query: {rewrited_query}
response: 
"""

ITERATION_FILTERING_PROMPT = """
You are an expert evaluator tasked with reviewing generated training data for AI model training. Your task is to verify the quality of each provided data example according to the following criteria:
Evaluate the provided data based on these criteria:

1. English Language:
 * Verify if all content in the data (conversation_history, query, rewrited_query, answer) is written exclusively in English.
 * If this criterion is not met, explain briefly under the key reason.

2. Pronoun Usage in Query:
 * Check if the query uses pronouns (he, she, it, this, that, etc.).
 * If pronouns are present, explicitly verify if the pronouns correctly and clearly refer to specific values present in either conversation_history or query.
 * Mark as "Pass" only if pronouns clearly reference specific values present in the provided context. Explicitly state exactly which specific value(s) the pronouns reference under reason.
 * Mark as "Fail" if pronouns are ambiguous or do not reference specific existing values.

3. Specific Value Consistency:
 * Check whether specific values in rewrited_query can be logically inferred or directly referenced from either conversation_history or the original query.
 * Identify and note any inconsistencies or unexplained additions explicitly under reason.

4. Pronoun Clarification in Rewrited Query:
 * Ensure that any pronouns used in query are clearly rewritten as specific values in rewrited_query.
 * Mark as "Pass" only if all pronouns from query have been explicitly replaced by clear and specific values in rewrited_query. If failed, explain the specific pronoun that was not clarified under reason.

5. API Rule Compliance:
 * Confirm that the parameters provided in answer.arguments correctly match the expected parameter types described for the respective API action.
 * For example, if a parameter expects a physical address but receives an email address, this constitutes a failure. Explicitly highlight any such discrepancies under reason.

Your response must be structured in JSON format:
{{
  "English Language": {{"status": "Pass" or "Fail", "reason": "If fail, describe why"}},
  "Pronoun Usage in Query": {{"status": "Pass" or "Fail", "reason": "Explicitly state exactly which specific value is referenced or why it failed"}},
  "Specific Value Consistency": {{"status": "Pass" or "Fail", "reason": "If fail, describe why"}},
  "Pronoun Clarification in Rewrited Query": {{"status": "Pass" or "Fail", "reason": "If fail, describe the pronoun that was not clarified"}},
  "API Rule Compliance": {{"status": "Pass" or "Fail", "reason": "If fail, describe why and specify the mismatched parameter"}}
}}

Carefully follow these instructions and ensure clarity and accuracy in your evaluation.
{data}
"""

ITERATION_SIMPLE_FILTERING_PROMPT = """
You are an expert evaluator tasked with reviewing generated training data for AI model training. Your task is to verify the quality of each provided data example according to the following criteria:
Evaluate the provided data based on these criteria:

1. English Language:
 * Verify if all content in the data (conversation_history, query, rewrited_query, answer) is written exclusively in English.
 * If this criterion is not met, explain briefly under the key reason.

2. API Rule Compliance:
 * Confirm that the parameters provided in answer.arguments correctly match the expected parameter types described for the respective API action.
 * For example, if a parameter expects a physical address but receives an email address, this constitutes a failure. Explicitly highlight any such discrepancies under reason.

Your response must be structured in JSON format:
{{
  "English Language": {{"status": "Pass" or "Fail", "reason": "If fail, describe why"}},  
  "API Rule Compliance": {{"status": "Pass" or "Fail", "reason": "If fail, describe why and specify the mismatched parameter"}}
}}

Carefully follow these instructions and ensure clarity and accuracy in your evaluation.
{data}
"""

USER_SEED_PROMPT = """
If the synthetic data generated using the LLM can be naturally modified to appear as if it was done by a human, please make the changes. The data consists of conversation_history, next_turn_query and rewrited_query.
The conversation_history includes the conversation between the user and assistant up until that point, with the user and assistant separated by "->". Each entity in the list represents one turn of the conversation.
The next_turn_query is the user’s next conversation (request) based on the conversation history, always referring to the first turn’s conversation history to make the next request.
What I would like you to do is review the next_turn_query and rewrited_query data to check if they sound natural. If any part sounds awkward, please revise it to make it more natural. The result should be provided in json format for next_turn_query and rewrited_query.

{tool}
"""

USER_NOISE_PROMPT = """
Generate a realistic user dialogue containing exactly one minor ASR-style error, selecting only one type from the following: a single typo, an omitted syllable, or a homophone substitution. Keep the dialogue natural and fluent; ensure the error is subtle enough to mimic typical speech-to-text inaccuracies.
Do not modify any specific values such as URLs, file names, or other exact references in the query.
The result should be provided in JSON format with two keys:

'new_query': containing the query with exactly one minor ASR-style error.
'diff': briefly describing which type of error was applied and the original value that was modified.
Do not include any additional keys.

{tool}
"""