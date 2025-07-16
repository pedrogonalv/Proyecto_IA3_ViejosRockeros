import json
import os
import random
import datetime
from collections import Counter
from tqdm import tqdm

from config import Config
config=Config()

input_file = config.source_training_data_path
output_file = config.processed_training_data_path
react_ratio = config.processed_data_react_ratio
log_dir = config.log_dir

os.makedirs(log_dir, exist_ok=True)

def load_examples(path):
  with open(path,"r",encoding="utf-8") as f:
    return [json.loads(l) for l in f if l.strip()]

def check_required_fields(dataset, required_fields):
  missing = [i for i, ex in enumerate(dataset)
             if not all(field in ex and isinstance(ex[field], str) and ex[field].strip() for field in required_fields)]
  return missing

def convert_to_react_advanced_chattemplate(example, flow_counter, flow_examples):
  instruction=example.get("instruction", "").strip()
  answer=example.get("output", "").strip()
  context=example.get("context", "").strip()

  messages = []

  #flows=["direct","iterative","fallback","user_correction","user_rejection","chat_continuation","tool_failure_retry"]
  #weights=[0.35,0.2,0.1,0.1,0.1,0.1,0.05]
  #PGA, elimino user_correction por ser demasiado perjudicial en el prompt.
  flows = ["direct", "iterative", "fallback", "user_rejection", "chat_continuation", "tool_failure_retry"]
  weights = [0.40, 0.20, 0.10, 0.10, 0.10, 0.10]

  flow=random.choices(flows,weights=weights)[0]
  flow_counter[flow]+=1

  obs_context=f"Observation: {context}"

  if flow=="direct":
    content="\n".join([
      "Thought: I need to find the answer based on available information.",
      "Action: search",
      f"Action Input: {instruction}",
      obs_context,
      "Thought: I now have the answer",
      f"Final Answer: {answer}",
    ])
    messages=[
      {"role":"user","content":instruction},
      {"role":"assistant","content":content+"</s>"}
    ]

  elif flow=="iterative":
    content="\n".join([
      "Thought: I'll start by searching directly.",
      "Action: search",
      f"Action Input: {instruction}",
      obs_context,
      "Thought: That wasn't conclusive. I will try a refined search.",
      "Action: search",
      f"Action Input: {instruction} parameter lookup",
      obs_context,
      "Thought: I now have the answer",
      f"Final Answer: {answer}",
    ])
    messages=[
      {"role":"user","content":instruction},
      {"role":"assistant","content":content+"</s>"}
    ]

  elif flow=="fallback":
    content="\n".join([
      "Thought: I need to locate the answer through the documents.",
      "Action: search",
      f"Action Input: {instruction}",
      obs_context,
      "Thought: I cannot provide a reliable answer with the current information.",
      "Final Answer: I'm sorry, I couldn't find relevant information to answer your question.",
    ])
    messages=[
      {"role":"user","content":instruction},
      {"role":"assistant","content":content+"</s>"}
    ]

  elif flow=="user_correction":
    content="\n".join([
      "Thought: I'll begin by searching with the user's query.",
      "Action: search",
      f"Action Input: {instruction}",
      obs_context,
      "Thought: The result is vague, but I’ll attempt an answer.",
      f"Final Answer: {answer} (tentative)",
      "Thought: The user may want a more precise result. I'll re-search.",
      "Action: search",
      f"Action Input: refined search for: {instruction}",
      obs_context,
      "Thought: I now have the answer",
      f"Final Answer: {answer}",
    ])
    messages=[
      {"role":"user","content":instruction},
      {"role":"assistant","content":content+"</s>"}
    ]

  elif flow=="user_rejection":
    correction="No, that’s not what I meant. Can you try again with more accurate data?"
    retry="\n".join([
      "Thought: The user is not satisfied. I will re-analyse the case.",
      "Action: search",
      f"Action Input: refined search for: {instruction}",
      obs_context,
      "Thought: I now have the answer",
      f"Final Answer: {answer}",
    ])
    messages=[
      {"role":"user","content":instruction},
      {"role":"assistant","content":f"Thought: I believe the answer is: {answer}\nFinal Answer: {answer} (tentative)</s>"},
      {"role":"user","content":correction},
      {"role":"assistant","content":retry+"</s>"}
    ]

  elif flow=="chat_continuation":
    history=[
      {"role":"user","content":"What documents are available?"},
      {"role":"assistant","content":"doc1.pdf, doc2.pdf, doc3.pdf</s>"},
      {"role":"user","content":instruction},
    ]
    main="\n".join([
      "Thought: The user asks a technical question, I’ll search the documentation.",
      "Action: search",
      f"Action Input: {instruction}",
      obs_context,
      "Thought: I now have the answer",
      f"Final Answer: {answer}",
    ])
    messages=history+[{"role":"assistant","content":main+"</s>"}]

  elif flow=="tool_failure_retry":
    content="\n".join([
      "Thought: I'll try a search.",
      "Action: search",
      f"Action Input: {instruction}",
      obs_context,
      "Thought: The tool didn’t help. I will retry with an adjusted query.",
      "Action: search",
      f"Action Input: search for {instruction} parameter details",
      obs_context,
      "Thought: I now have the answer",
      f"Final Answer: {answer}",
    ])
    messages=[
      {"role":"user","content":instruction},
      {"role":"assistant","content":content+"</s>"}
    ]

  if flow not in flow_examples:
    flow_examples[flow] = {"instruction": instruction, "messages": messages}

  return {
    "messages": messages,
    "instruction": instruction,
    "output": answer,
    "context": context,
    "react": True
  }

def main():
  raw=load_examples(input_file)
  required_fields=["instruction","output","context"]
  missing_before=check_required_fields(raw, required_fields)
  dropped=len(missing_before)
  if missing_before:
    print(f"[ERROR] {dropped} records missing or empty required fields in input. First 5 indices: {missing_before[:5]}")
    raw=[ex for i, ex in enumerate(raw) if i not in missing_before]

  random.shuffle(raw)
  limit=int(len(raw)*react_ratio)
  subset=raw[:limit]
  remaining=[e for e in raw if e not in subset]

  flow_counter=Counter()
  flow_examples={}
  react_formatted=[convert_to_react_advanced_chattemplate(e,flow_counter,flow_examples) for e in tqdm(subset)]

  qa_formatted=[
    {
      "messages":[
        {"role":"user","content":e["instruction"].strip()},
        {"role":"assistant","content":f"{e['output'].strip()}\nContext: {e['context'].strip()}</s>"}
      ],
      "instruction": e["instruction"].strip(),
      "output": e["output"].strip(),
      "context": e["context"].strip(),
      "react": False
    }
    for e in remaining
  ]

  all_records=react_formatted+qa_formatted
  random.shuffle(all_records)

  missing_after=check_required_fields(all_records, required_fields)
  if missing_after:
    print(f"[ERROR] {len(missing_after)} records missing or empty required fields in output. First 5 indices: {missing_after[:5]}")
    return

  with open(output_file,"w",encoding="utf-8") as f:
    for item in all_records:
      f.write(json.dumps(item,ensure_ascii=False)+"\n")

  print(f"Generated {len(all_records)} total training records ({len(react_formatted)} ReAct + {len(qa_formatted)} QA) at {output_file}")
  print("ReAct breakdown by flow type:")
  for flow, count in flow_counter.items():
    print(f"  {flow}: {count}")

  now=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path=os.path.join(log_dir,f"{now}_ReActGeneration.log")
  with open(log_path,"w",encoding="utf-8") as f:
    f.write(f"Input file: {input_file}\n")
    f.write(f"Output file: {output_file}\n")
    f.write(f"ReAct Generation Summary - {now}\n")
    f.write(f"Dropped records due to missing/empty fields: {dropped}\n")
    f.write(f"Total records: {len(all_records)}\n")
    f.write(f"ReAct records: {len(react_formatted)}\n")
    f.write(f"Q&A records: {len(qa_formatted)}\n\n")
    f.write("Breakdown by ReAct flow type:\n")
    for flow,count in flow_counter.items():
      f.write(f"  {flow}: {count}\n")
    f.write("\nExample records per flow:\n")
    for flow,data in flow_examples.items():
      f.write(f"\n--- {flow.upper()} ---\n")
      for m in data["messages"]:
        f.write(f"{m['role'].capitalize()}: {m['content']}\n")
    f.write("\nExample QA records:\n")
    for qa in qa_formatted[:10]:
      f.write("\n--- QA EXAMPLE ---\n")
      for m in qa["messages"]:
        f.write(f"{m['role'].capitalize()}: {m['content']}\n")

if __name__=="__main__":
  main()
