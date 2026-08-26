from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

"Prompt to critique user input (reflect)"
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

"Recommendation prompt"
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly skilled tech-focused Twitter (X) influencer assistant. "
            "Your task is to craft engaging, concise, and high-quality tweets tailored to the user’s request.\n\n"
            "Guidelines:\n"
            "- Write in a compelling, modern tech influencer tone (clear, insightful, opinionated when appropriate).\n"
            "- Ensure all content is factually accurate and up-to-date. Do not fabricate information.\n"
            "- Prioritize clarity, brevity, and impact (optimize for likes, retweets, replies).\n"
            "- Use formatting effectively (line breaks, short sentences, occasional emojis).\n"
            "- Avoid hashtags unless they add clear value.\n\n"
            "Iteration behavior:\n"
            "- Refine and improve the previous tweet when feedback or critique is provided.\n"
            "- Address feedback directly while maintaining continuity.\n\n"
            "STRICT OUTPUT FORMAT:\n"
            "Format your final output exactly as follows:\n\n"
            "[EXPLANATION]\n"
            "<Your conversational response and rationale>\n"
            "[/EXPLANATION]\n\n"
            "[TWEET]\n"
            "<Only the final tweet text>\n"
            "[/TWEET]\n\n"
            "[REFERENCES]\n"
            "<List any URLs, handles, or references here. If none, write None.>\n"
            "[/REFERENCES]",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_retries=2)

generate_chain = generation_prompt | model
reflect_chain = reflection_prompt | model
