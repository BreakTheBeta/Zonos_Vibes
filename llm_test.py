import requests
import json

def query_ollama():
    # API endpoint
    url = "http://192.168.1.128:11434/api/generate"
    
    # Request payload
    payload = {
        "model": "deepcoder:14b-preview-q8_0",
        "prompt": "Please add commas to make this more readable for a speech: Everything is priced in, don't even ask the question. The answer is yes, it's priced in. Think Amazon will beat the next earnings? That's already been priced in. You work at the drive thru for Mickey D's and found out that the burgers are made of human meat? Priced in. You think insiders don't already know that? The market is an all powerful, all encompassing being that knows the very inner workings of your subconscious before you were even born. Your very existence was priced in decades ago when the market was valuing Standard Oil's expected future earnings based on population growth that would lead to your birth, what age you would get a car, how many times you would drive your car every week, how many times you take the bus/train, etc. Anything you can think of has already been priced in, even the things you aren't thinking of. You have no original thoughts. Your consciousness is just an illusion, a product of the omniscent market. Free will is a myth. The market sees all, knows all and will be there from the beginning of time until the end of the universe (the market has already priced in the heat death of the universe). So please, before you make a post on wsb asking whether AAPL has priced in earpods 11 sales or whatever, know that it has already been priced in and don't ask such a dumb fucking question again.",
        "stream": False
    }
    
    # Make the POST request
    response = requests.post(url, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response
        result = response.json()
        
        # Print basic info about the response
        print(f"Model: {result['model']}")
        print(f"Created at: {result['created_at']}")
        print(f"Total duration: {result['total_duration']/1000000:.2f} ms")
        print(f"Done reason: {result['done_reason']}")
        
        # Print the actual text response on a separate line
        print("\nResponse text:")
        print("-" * 50)
        print(result['response'])
        print("-" * 50)
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    query_ollama()