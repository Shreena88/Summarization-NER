from transformers import pipeline
import re

# Load your local summarization model
summarizer = pipeline(
    "summarization",
    model=r"D:/1.SHREEE/Summarization+NER/utils/model_directory",
    tokenizer=r"D:/1.SHREEE/Summarization+NER/utils/model_directory"
)

def clean_dialogue(text):
    """Remove speaker tags like 'Neha:' or 'Arjun:' for cleaner summarization"""
    return re.sub(r"^\w+: ", "", text, flags=re.MULTILINE)

def summarize_text(text, max_length=270, min_length=70):
    """Summarizes dialogue text and returns a summary string"""
    cleaned_text = clean_dialogue(text)

    num_words = len(cleaned_text.split())
    
    min_len = max(20, int(num_words * 0.2))  # at least 20 tokens
    max_len = max(50, int(num_words * 0.4))  # at least 50 tokens

    summary = summarizer(
        cleaned_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        truncation=True
    )
    return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    dialogue = """
    Neha: Hey Arjun! I can’t believe I ran into you here at Starbucks. It’s been years!
    Arjun: Neha! Wow, what a surprise. I just came back from London last week. How are you doing?
    Neha: I’m doing well. I’m still working at Infosys in Bangalore. What about you?
    Arjun: I joined Google in London last year. The work is intense but exciting.
    Neha: That sounds amazing. Do you remember Riya from our college?
    Arjun: Of course, Riya Sharma! We used to study together for the computer networks exam. Where is she now?
    Neha: She moved to New York. She’s working as a data scientist at IBM.
    Arjun: Wow, that’s great! By the way, did you hear about Rahul?
    Neha: Rahul Mehta? Yes, I met him last month in Delhi. He’s now running his own startup.
    Arjun: That’s so typical of Rahul. Always full of ideas.
    Neha: Haha, true. Oh, by the way, I saw Professor Verma recently. He asked about you.
    """

    print("----- Summary -----")
    print(summarize_text(dialogue))
