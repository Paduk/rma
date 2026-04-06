# filtering api file for debugging
import pdb
import os
import json

def s3_filter():
    target = ["ACTION_VIEW_CONTACT - dial", "ACTION_EDIT_CONTACT - search_location", "ACTION_EDIT_CONTACT - ACTION_NAVIGATE_TO_LOCATION", "ACTION_INSERT_CONTACT - search_location", "ACTION_EDIT_CONTACT - dial", "get_contact_info - send_email", "get_contact_info - search_location", "get_contact_info - ACTION_NAVIGATE_TO_LOCATION", "ACTION_INSERT_CONTACT - send_email", "get_contact_info - dial", "ACTION_INSERT_CONTACT - dial", "ACTION_INSERT_EVENT - search_location", "ACTION_INSERT_EVENT - ACTION_NAVIGATE_TO_LOCATION", "get_contact_info - send_message"]

    prev_plans = []
    for t in target:
        prev_plan = t.split(' - ')[0]
        prev_plans.append(prev_plan)

    apis = []
    with open("api_v3.0.1.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if data["name"] in prev_plans:
                        apis.append(data)

    print(len(apis))
    for api in apis:
        plan = api["name"]
        if "new_next_turn_functions" not in api:
            api["new_next_turn_functions"] = []
        for t in target:
            prev_plan = t.split(' - ')[0]
            next_plan = t.split(' - ')[1]
            if prev_plan == plan:
                for ne in api["next_turn_functions"]:
                    if ne["function_name"] == next_plan:
                        api["new_next_turn_functions"].append(ne)

    OUTPUT_FILE = "s3_apis.jsonl"
    output_file = open(OUTPUT_FILE, "w")                    
    for api in apis:
        api.pop("next_turn_functions")                    
        api["next_turn_functions"] = api["new_next_turn_functions"]
        api.pop("new_next_turn_functions")
        output_file.write(json.dumps(api, ensure_ascii=False)+"\n")
        output_file.flush()        

    #pdb.set_trace()

