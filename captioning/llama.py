from transformers import pipeline
import torch
import json

frame_caption_file = '../captions/hmdb51_min0max7710f_captions.json'
with open(frame_caption_file) as f:
    frame_captions = json.load(f)
    
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

classes = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]
count = 0
prompt = "Given the following 51 classes of actions: " + ' '.join(classes) + ", which one might be occuring" + \
    "in a video composed of 10 evenly sampled frames with the following captions: "
for video, captions in frame_captions.items():
    if count > 10:
        break
    messages = [{
        "role": "user",
        "content": prompt + ' '.join(captions) + ". Answer in one word, consisting of just your answer of what action it is."
    }]
    outputs = pipe(
        messages,
        max_new_tokens=77,
        do_sample=False,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(assistant_response)
    count += 1

