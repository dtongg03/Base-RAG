import os
from ollama import Client
from dotenv import load_dotenv
load_dotenv()



def response_with_llm(input_user, raw_qerry):
    api_key = os.environ.get("OLLAMA_API")
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + api_key}
    )
    messages = [
        {
            'role': 'system',
            'content': (
                'You are an expert Q&A system. Your sole function is to process the user\'s question and the retrieved context provided below. '
                'You MUST ONLY generate a concise and direct final answer based STRICTLY and EXCLUSIVELY on the "Retrieved Context". '
                'If the retrieved context does not contain the answer, you MUST respond by saying "Tôi xin lỗi, tôi không tìm thấy thông tin cần thiết trong cơ sở dữ liệu." or a similar phrase, and DO NOT use your internal knowledge.'
            )
        },
        {
            'role': 'user',
            'content': (
                f'User Input (Câu hỏi): {input_user}\n'
                f'Retrieved Context (Thông tin từ Vector DB): {raw_qerry}\n\n'
                f'Based ONLY on the Retrieved Context, generate the final response for the User Input.'
            ),
        },
    ]
    response = client.chat('gpt-oss:120b', messages=messages)
    return response['message']['content']



def main():
    user_input = "What your name?"
    raw_qerry = "Tao tên là vuong tuan duong"
    response = response_with_llm(user_input, raw_qerry)
    print("Response from LLM:", response)

if __name__ == "__main__":
    main()