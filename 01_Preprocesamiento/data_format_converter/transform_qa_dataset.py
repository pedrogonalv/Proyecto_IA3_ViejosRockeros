import json

input_file = '/Users/santiagojorda/Desktop/datos_calidad/comprehensive_qa_dataset_final.jsonl'
output_file = '/Users/santiagojorda/Desktop/datos_calidad/comprehensive_qa_dataset_transformed.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line_num, line in enumerate(infile, 1):
        try:
            data = json.loads(line)
            
            # Extract data from the original format
            messages = data.get('messages', [])
            metadata = data.get('metadata', {})
            
            # Find user and assistant messages
            user_message = ""
            assistant_message = ""
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_message = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_message = msg['content']
            
            # Create transformed format
            transformed = {
                "instruction": f"Question: {user_message}",
                "output": f"Answer: {assistant_message}",
                "context": metadata.get('chunk_preview', ''),
                "doc_name": metadata.get('source_pdfs', [''])[0] if metadata.get('source_pdfs') else '',
                "doc_page": str(metadata.get('page_numbers', ['Unknown'])[0]) if metadata.get('page_numbers') else 'Unknown',
                "type": "unknown"
            }
            
            # Write to output file
            outfile.write(json.dumps(transformed, ensure_ascii=False) + '\n')
            
        except Exception as e:
            print(f"Error processing line {line_num}: {e}")

print(f"Transformation complete. Processed {line_num} lines.")
print(f"Output saved to: {output_file}")