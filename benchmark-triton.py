import asyncio
import json
import random
import time
from typing import List
from tritonclient.http.aio as httpclient
import aiohttp
import numpy as np
from tabulate import tabulate

# Specify the Triton server URL and model name
url = "localhost:8000"  # Adjust if Triton server is running on a different machine or port
model_name = "e5_base"


passages_300 = [

  "Engaging in daily physical activities such as walking, jogging, swimming,promotes cardiovascular health, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. Outdoor activities like hiking or biking provide both physical and mental stimulation, increasing your exposure to nature and natural sunlight, which boosts Vitamin D production. Additionally, group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}. This is based on the latest health guidelines, which suggest incorporating balance of aerobic. The recommended amount of {entity} for {person} is around {amount}. Engaging in daily physical activities such as walking, jogging, swimming, promotes cardiovascular health, improves lung capacity, and enhances muscle strength.",

  "Engaging in daily physical activities such as walking, jogging, swimming  promotes cardiovascular health, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. Outdoor activities like hiking or biking provide both physical and mental stimulation, increasing your exposure to nature and natural sunlight, which boosts Vitamin D production. Additionally, group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}. This is based on the latest health guidelines, which suggest incorporating balance of aerobic and strength-training exercises to maximize health benefits.Without proper {action}, you might experience {symptom}. Staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout.",

  "Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Additionally, staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion.  Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion.  Learning {topic} is crucial in today's {industry}.",

  "It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function.  Activities such as hiking, cycling, and outdoor yoga are particularly beneficial. Incorporating mindfulness practices, such as deep breathing or meditation during outdoor workouts, can further amplify the mental health benefits, allowing individuals to manage stress more effectively. Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}. As you advance, delve deeper into complex ideas such as {beginner_concepts}. Attending workshops or online classes can enhance your learning process, enabling you to interact with experts and peers who share the same interests, and providing you with the resources needed for long-term success in your field. Continuous self-evaluation and application of knowledge gained can significantly accelerate your mastery of {topic}. Delve deeper into complex ideas such as {beginner_concepts}. Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}.Delve deeper into complex ideas such as {beginner_concepts}."

]


passages_500 = [
"Can you explain {topic} to a beginner? You can use any help like AI. It also helps to regulate weight by burning excess calories and increases energy levels. The importance of regular physical exercise cannot be overstated in maintaining overall health and well-being. Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Additionally, staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion. Adequate rest and sleep are equally vital for recovery, allowing your muscles to heal and grow stronger over time. Regular exercise after a certain age is vital for muscles as it prevents atrophy and weakness. It ensures that as people age, they maintain functional mobility, reducing the likelihood of falls and injuries. Cardiovascular exercises, combined with strength training, are particularly effective for this. More and more, experts recommend stretching exercises like yoga and Pilates to improve flexibility and posture. These exercises reduce the likelihood of strain and injury, keeping joints flexible and improving overall physical alignment. For older adults, staying physically active becomes one of the most critical factors in maintaining independence and quality of life. Regular exercise also improves balance, coordination, and mental alertness, which all contribute to healthier aging. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}.",
" It also helps to regulate weight by burning excess calories and increases energy levels. Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. Outdoor activities like hiking or biking provide both physical and mental stimulation, increasing your exposure to nature and natural sunlight, which boosts Vitamin D production.Group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}. This is based on the latest health guidelines, which suggest incorporating a balance of aerobic and strength-training exercises to maximize health benefits. Exercise is great, but it’s also crucial to ensure you’re practicing proper form and techniques to avoid injuries. Strengthening your core is another vital component often overlooked in many exercise routines. A strong core improves posture, reduces back pain, and makes daily activities, such as bending or lifting, easier and safer. Incorporating core-focused exercises such as planks, bridges, and sit-ups into your regular routine can significantly enhance overall fitness. Don't forget to cool down after exercise with proper stretches to maintain flexibility. Additionally, group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Can you explain {topic} to a beginner? You can use any help like AI",
" Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors can improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion. Adequate rest and sleep are equally vital for recovery, allowing your muscles to heal and grow stronger over time. The exercise should also help one recover from surgery. It also prevents aging, allowing individuals to maintain their vitality and mobility well into their later years. For people recovering from injuries or surgeries, physical rehabilitation through light exercise can aid in a quicker and more effective recovery. Regular movement prevents stiffness in joints and helps restore strength. Beyond physical health, exercise nurtures mental resilience. Do regular exercise when you are young. The importance of regular physical exercise cannot be overstated in maintaining overall health and well-being. Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Attending workshops or online classes can enhance your learning process. Can you explain {topic} to a beginner? You can use any help like AI",
"Can you explain {topic} to a beginner? You can use any help like AI .The Education and Learning {topic} is crucial in today's {industry}. As you advance, delve deeper into complex ideas such as {beginner_concepts}. Attending workshops or online classes can enhance your learning process, enabling you to interact with experts and peers who share the same interests, providing you with the resources needed for long-term success in your field. Continuous self-evaluation and application of knowledge gained can significantly accelerate your mastery of {topic}. Additionally, fitness programs often include mindfulness practices, helping individuals balance both mental and physical well-being. Many people find that adopting breathing techniques during exercise significantly enhances their performance, helping them to pace themselves more effectively. Consult the doctor before doing any hard exercises. The importance of regular physical exercise cannot be overstated in maintaining overall health and well-being. Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Activities such as hiking, cycling, and outdoor yoga are particularly beneficial. Incorporating mindfulness practices, such as deep breathing or meditation during outdoor workouts, can further amplify the mental health benefits, allowing individuals to manage stress more effectively. Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}. As you advance, delve deeper into complex ideas such as {beginner_concepts}. The Online classes can enhance your learning process, enabling you to interact with experts and peers who share the same interests, providing you with the resources needed for long-term success in your field. Continuous self-evaluation and application of knowledge gained can significantly accelerate your mastery of {topic}. Additionally, fitness programs often include mindfulness practices, helping individuals balance both mental and physical well-being."

] 
"""






passages_300 = [
"Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Additionally, staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion.  Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}.",

"Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. Outdoor activities like hiking or biking provide both physical and mental stimulation, increasing your exposure to nature and natural sunlight, which boosts Vitamin D production. Additionally, group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}. This is based on the latest health guidelines" ,

"Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Additionally, staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion.  Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}.",

"It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. Exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function.  Activities such as hiking, cycling, and outdoor yoga are particularly beneficial. Incorporating mindfulness practices, such as deep breathing or meditation during outdoor workouts, can further amplify the mental health benefits, allowing individuals to manage stress more effectively. Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}. As you advance, delve deeper into complex ideas such as {beginner_concepts}. Attending workshops or online classes can enhance your learning process, enabling you to interact with experts and peers who share the same interests, providing you with the resources needed for long-term success in your field. Continuous self-evaluation and application of knowledge gained can significantly accelerate your mastery of {topic}. Delve deeper into complex ideas such as {beginner_concepts}."
]


passages_500 = [
"What is the process of making  {dish} in a country like Russia where weather is cold and rainy? It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Additionally, staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion. Regular exercise after a certain age is vital for muscles as it prevents atrophy and weakness. It ensures that as people age, they maintain functional mobility, reducing the likelihood of falls and injuries. Cardiovascular exercises, combined with strength training, are particularly effective for this. More and more, experts recommend stretching exercises like yoga and Pilates to improve flexibility and posture. These exercises reduce the likelihood of strain and injury, keeping joints flexible and improving overall physical alignment. For older adults, staying physically active becomes one of the most critical factors in maintaining independence and quality of life. Regular exercise also improves balance, coordination, and mental alertness, which all contribute to healthier aging. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Work harder daily.",

"what is the process of making the {dish} in a country like India where weather is hot and rainy? It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. Outdoor activities like hiking or biking provide both physical and mental stimulation, increasing your exposure to nature and natural sunlight, which boosts Vitamin D production. Additionally, group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}. This is based on the latest health guidelines, which suggest incorporating a balance of aerobic and strength-training exercises to maximize health benefits. Exercise is great, but it’s also crucial to ensure you’re practicing proper form and techniques to avoid injuries. Strengthening your core is another vital component often overlooked in many exercise routines. A strong core improves posture, reduces back pain, and makes daily activities, such as bending or lifting, easier and safer. Incorporating core-focused exercises such as planks, bridges, and sit-ups into your regular routine can significantly enhance overall fitness. Don't forget to cool down after exercise with proper stretches to maintain flexibility.  Additionally, group exercises, such as yoga classes or team sports, foster a sense of community and social interaction, which is critical in reducing feelings of loneliness and improving overall life satisfaction. The recommended amount of {entity} for {person} is around {amount}.",

"What is the process of making {dish} in a country like America where weather is cold. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature, reduce feelings of isolation, and breathe fresh air. {action} is essential for maintaining {object}. Without proper {action}, you might experience {symptom}. Additionally, staying hydrated and consuming a balanced diet that supports your physical efforts will further enhance the effectiveness of your workout. Supplements like vitamins, minerals, and protein shakes can also support your body’s needs during periods of high physical exertion. Allowing your muscles to heal and grow stronger over time. The exercise should also help one recover from surgery. It also prevents aging, allowing individuals to maintain their vitality and mobility well into their later years. For people recovering from injuries or surgeries, physical rehabilitation through light exercise can aid in a quicker and more effective recovery. Regular movement prevents stiffness in joints and helps restore strength. Beyond physical health, exercise nurtures mental resilience. The importance of regular physical exercise cannot be overstated in maintaining overall health and well-being. Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. Furthermore, exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Exercising outdoors, in particular, can further improve mental well-being by allowing individuals to connect with nature",

"what is the process of making {dish} in a country like China where weather is cold? Continuous self-evaluation and application of knowledge gained can significantly accelerate your mastery of {topic}. Additionally, fitness programs often include mindfulness practices, helping individuals balance both mental and physical well-being. Many people find that adopting breathing techniques during exercise significantly enhances their performance, helping them to pace themselves more effectively. Consult the doctor before doing any hard exercises. The importance of regular physical exercise cannot be overstated in maintaining overall health and well-being. Engaging in daily physical activities such as walking, jogging, swimming, or cycling promotes cardiovascular health, improves lung capacity, and enhances muscle strength. It also helps to regulate weight by burning excess calories and increases energy levels, making individuals feel more alert and productive. Exercise stimulates the release of endorphins, the body's natural mood elevators, which reduce stress and anxiety levels. Studies have shown that regular exercise can reduce the risk of chronic diseases such as heart disease, diabetes, and hypertension. In addition to the physical benefits, exercise plays a significant role in improving mental health by boosting self-esteem, reducing symptoms of depression, and enhancing cognitive function. Activities such as hiking, cycling, and outdoor yoga are particularly beneficial. Incorporating mindfulness practices, such as deep breathing or meditation during outdoor workouts, can further amplify the mental health benefits, allowing individuals to manage stress more effectively. Learning {topic} is crucial in today's {industry}. To begin, focus on {beginner_concepts}. As you advance, delve deeper into complex ideas such as {beginner_concepts}. Attending workshops or online classes can enhance your learning process, enabling you to interact with experts and peers who share the same interests, providing you with the resources needed for long-term success in your field. Continuous self-evaluation and application of knowledge gained can significantly accelerate your mastery of {topic}. Delve deeper into complex ideas such as {beginner_concepts}."

]
"""

# Synthetic data generation functions
def generate_synthetic_query_input(num_queries: int) -> List[str]:
    """Generates synthetic query input text with controlled token size."""
    queries = [
        "how much {entity} does {person} need in a day once in a while?",   
        "what is the process of making {dish} in a country like North Korea?",
        "why is {concept} so important for {profession}, is it necessary to learn it?",
        "how does {action} affect {object}? Explain in brief without making judgements about it.", 
        "can you explain {topic} to a beginner? You can use any help like AI tools" 
    ]
    
    synthetic_queries = []
    entities = ["protein", "vitamin D", "water", "calories"]
    persons = ["a female", "a male", "an athlete", "a child"]
    dishes = ["pumpkin soup", "pasta", "salad", "fried rice"]
    concepts = ["exercise", "good nutrition", "hydration", "sleep"]
    professions = ["athletes", "students", "software engineers", "teachers"]
    actions = ["drinking water", "eating", "exercise"]
    objects = ["muscles", "brain", "bones", "energy levels"]
    topics = ["nutrition", "exercise", "coding", "data science"]

    for _ in range(num_queries):
        template = random.choice(queries)
        query = template.format(
            entity=random.choice(entities),
            person=random.choice(persons),
            dish=random.choice(dishes),
            concept=random.choice(concepts),
            profession=random.choice(professions),
            action=random.choice(actions),
            object=random.choice(objects),
            topic=random.choice(topics)
        )
        query_text = f"query: {query}" 
        synthetic_queries.append(query_text)
        
    return synthetic_queries

def generate_synthetic_passage_input(passages: List[str], num_passages: int) -> List[str]:
    """Generates synthetic passage input text with controlled token size."""
    synthetic_passages = []
    entities = ["protein", "vitamin D", "water", "calories"]
    persons = ["a female", "a male", "an athlete", "a child"]
    amounts = ["46 grams per day", "600 IU", "8 cups", "2000 calories"]
    dishes = ["pumpkin soup", "pasta", "salad", "fried rice"]
    ingredients = ["pumpkin", "pasta", "olive oil", "garlic", "onions"]
    concepts = ["good nutrition", "hydration", "sleep", "exercise"]
    goals = ["high performance", "mental clarity", "optimal health"]
    reasons = ["better focus", "increased energy", "improved muscle recovery"]
    actions = ["drinking water", "eating healthy", "sleeping enough"]
    objects = ["energy levels", "muscles", "bones", "brain"]
    symptoms = ["fatigue", "muscle cramps", "headaches"]
    topics = ["coding", "nutrition", "exercise"]
    industries = ["tech industry", "fitness world", "education sector"]
    beginner_concepts = ["basics of syntax", "functions and variables", "data structures"]

    for _ in range(num_passages):
        template = random.choice(passages)
        passage = template.format(
            entity=random.choice(entities),
            person=random.choice(persons),
            amount=random.choice(amounts),
            dish=random.choice(dishes),
            ingredients=", ".join(random.sample(ingredients, 3)),
            profession=random.choice(["athletes", "teachers", "engineers"]),
            concept=random.choice(concepts),
            goal=random.choice(goals),
            reason=random.choice(reasons),
            action=random.choice(actions),
            object=random.choice(objects),
            symptom=random.choice(symptoms),
            topic=random.choice(topics),
            industry=random.choice(industries),
            beginner_concepts=random.choice(beginner_concepts)
        )
        passage_text = f"passage: {passage}"
        synthetic_passages.append(passage_text)
        
    return synthetic_passages

async def get_embeddings(input_texts: List[str]) -> float:
    """Asynchronously fetch embeddings and return elapsed time."""  
    start_time = time.perf_counter()
    inputs = []
    input = httpclient.InferInput("text", [len(input_texts),], "BYTES")
    input.set_data_from_numpy(np.array(input_texts, dtype=object), binary_data=False)
    inputs.append(input)
    
    outputs = []
    triton_client = httpclient.InferenceServerClient(
                url="localhost:8000", verbose=False
            )
    results = await triton_client.infer(
        "ensemble_model",
        inputs,
        outputs=outputs
    )
    embeddings = results.as_numpy("embeddings")
    print(embeddings.shape)
      
    elapsed_time = (time.perf_counter() - start_time) * 1000 

    
    return elapsed_time

async def run_hug_e5(batch_size: int, input_type: str, semaphore: asyncio.Semaphore) -> float:
    """Run the huggingface model E5 and measure execution time."""
    async with semaphore:
        queries = generate_synthetic_query_input(batch_size)

        print(f"Processing batch of size {len(queries)}")
        elapsed_time = await get_embeddings(queries, input_type)
        print(f"Execution finished in {elapsed_time:.2f} ms")
        return elapsed_time

# Benchmark runner
async def run_benchmark(number_of_runs: int, max_workers: int, batch_size: int, input_type: str, input_tokens: int, table_data: List[List[str]]):
    """Run benchmark with specified parameters."""
    latencies = []
    semaphore = asyncio.Semaphore(max_workers)  # Limit concurrency


    tasks = [
            asyncio.create_task(run_hug_e5(batch_size, input_type, input_tokens, semaphore))
            for _ in range(number_of_runs * max_workers)
        ]

    for task in asyncio.as_completed(tasks):
            try:
                latency = await task
                latencies.append(latency)
            except Exception as e:
                print(f"Task encountered an exception: {e}")

    if not latencies:
        print("No latencies recorded.")
        return

    # Calculate required metrics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p25_latency, p50_latency, p75_latency, p90_latency, p95_latency, p99_latency = np.percentile(
        latencies, (25, 50, 75, 90, 95, 99))
    
    throughput = (max_workers / (avg_latency / 1000)) if avg_latency > 0 else 0   # Inputs per second

    table_data.append([
        "Request Latency (ms)", input_type, input_tokens, batch_size, max_workers,
        f"{avg_latency:.2f}", f"{min_latency:.2f}", f"{max_latency:.2f}",
        f"{p99_latency:.2f}", f"{p95_latency:.2f}", f"{p90_latency:.2f}",
        f"{p75_latency:.2f}", f"{p50_latency:.2f}", f"{p25_latency:.2f}",
        f"{throughput:.2f}"
    ])

# Main benchmarking function
async def start_benchmarking():
    headers = ["Metric", "Input Type", "Input Tokens", "Batch Size", "Concurrency", "Avg (ms)", "Min (ms)", 
               "Max (ms)", "P99 (ms)", "P95 (ms)", "P90 (ms)", "P75 (ms)", "P50 (ms)", "P25 (ms)", 
               "Throughput (inputs/s)"]
    
    table_data = []

    benchmark_data = [
        (200, 1, 64, 'passage', 300),
        (200, 3, 64, 'passage', 300),
        (200, 5, 64, 'passage', 300),
        (200, 1, 64, 'passage', 500),
        (200, 3, 64, 'passage', 500),
        (200, 5, 64, 'passage', 500),
        (200, 1, 1, 'query', 20),
        (200, 3, 1, 'query', 20),
        (200, 5, 1, 'query', 20),
        (200, 7, 1, 'query', 20),
        (200, 11, 1, 'query', 20),
        (200, 13, 1, 'query', 20),
        (200, 15, 1, 'query', 20)
    ]

   

    for number_of_runs, max_workers, batch_size, input_type, input_tokens in benchmark_data:
        print(f"Running benchmark: {input_type}, {input_tokens} tokens, batch size {batch_size}, concurrency {max_workers}")
        await run_benchmark(number_of_runs, max_workers, batch_size, input_type, input_tokens, table_data)
        print(f"Benchmarking completed for {input_type} with {input_tokens} input tokens, {max_workers} concurrency, and {batch_size} batch size.")

    # Generate HTML table
    html_table = tabulate(table_data, headers=headers, tablefmt="html")
    print(html_table)

if __name__ == "__main__":
    asyncio.run(start_benchmarking())

