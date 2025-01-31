import re
import os
import sys
import json
import torch
import time
import spacy
import openai

import numpy as np

from openai import OpenAI
from typing import List
from PIL import Image
from transformers import AutoTokenizer 
sys.path.append("path/to/LLaVA")                           # LLaVA code path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

model_path = "path/to/liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    # load_8bit=True,
)
print("Load LLaVA-v1.5-7b Done.")


NUM_SECONDS_TO_SLEEP = 0.5
client = OpenAI(
      api_key="sk-xxxxxxxxxxxxxxxx"
)
print("Load LLM Done.")


# --------------------- prompt design ---------------------
PROMPT_TEMPLATE = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: {question}
AI: '''

PROMPT_HISTORY_TEMPLATE = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
{history}
Human: {question}
AI: '''



PROMPT_TEMPLATE_PREPROCESS = """
Given a passage, you are required to replace pronouns such as "they" with the actual entities they refer to based on the context, then output the passage after replacement.
Only replace the pronouns if there is any, and do not change anything else in the original passage. 
If there is nothing to replace, then keep the original sentences unchanged, do not add any new sentence.
The modification should be as small as possible, and the output passage should have the same number of sentences as the original passage.

Examples:
Passage:
The image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. He appears to be smoking.

Rewritten passage:
The image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. The man appears to be smoking.

Passage:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed. The dog is in the center of the group, while the cats are positioned to the left and right. 

Rewritten passage:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed. The dog is in the center of the group, while the cats are positioned to the left and right. 


Please only output the passage, without any explanation.
Passage:
{text}

Rewritten passage:
"""


PROMPT_TEMPLATE_TARGET = """"
You are given a sentence, extract the entities within the sentence for me. 
[Task]
Your task is to extract the common objects and summarize them as general categories without repetition, merging essentially similar objects.
Avoid extracting abstract or non-specific entities. 
Extract entity in the singular form. Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None".
DO NOT RESPOND WITH ANYTHING ELSE.

Here are examples:
[Sentence]:
The image depicts a man laying on the ground next to a motorcycle, which appears to have been involved in a crash.
[Response]:
man.motorcycle
[Sentence]:
There are a few people around, including one person standing close to the motorcyclist and another person further away.
[Response]:
person.motorcyclist
[Sentence]:
No, there is no car in the image.
[Response]:
car
[Sentence]:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed.
[Response]:
dog.cat.bed

Now complete the following: 
[Sentence]:
{sent}
[Response]:
"""


PROMPT_TEMPLATE_FACTOR_ATTRIBUTE = """
You will receive a piece of text that describes an object, and the given object.
[Task]
Your task is to accurately identify and extract every attribute associated with the given object in the provided text. Each claim should be concise (less than 15 words) and self-contained, corresponding to only one attribute. 
You MUST only respond in the format as required. Each line should contain the original claim and the modified claim with all the mentions of the given object being replaced with "the object". 
DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED.

[Response Format]
original claim&modified claim

Here are examples:
[Text]:
The man is wearing a baseball cap and appears to be smoking.
[Entity]:
man
[Response]:
The man is wearing a baseball cap.&The object is wearing a baseball cap.
The man appears to be smoking.&The object appears to be smoking.
[Text]:
The snowboard in the picture is a red and white snowboard, which is being ridden by a person on a snowy slope. The snowboarder is wearing a helmet and is in the middle of the slope, likely enjoying the thrill of skiing down the mountain.
[Entity]:
snowboard
[Response]:
The snowboard in the picture is red and white.&The object in the picture is red and white.
The snowboard is being ridden by a person on a snowy slope.&The object is being ridden by a person on a snowy slope.
[Text]:
The truck in the picture is a red and black pickup truck, which is parked in a driveway. It is a small truck, likely a compact or mid-size model, and appears to be in good condition.
[Entity]:
truck
[Response]:
The truck is red and black.&The object is red and black.
The truck is a pickup truck.&The object is a pickup object.
The truck is parked in a driveway.&The object is parked in a driveway.
The truck is a small truck.&The object is a small object.
The truck appears to be in good condition.&The object appears to be in good condition.
[Text]:
The dining table in the image is a wooden table with a red and black checkered tablecloth.
[Entity]:
dining table
[Response]:
The dining table in the image is made of wood.&The object in the image is made of wood.
The dining table has a red and black checkered tablecloth.&The object has a red and black checkered tablecloth.


Now complete the following: 
[Text]:
{sent}
[Entity]:
{entity}
[Response]:
"""


PROMPT_TEMPLATE_REPHRASE_TORTURE_2 = """
You will receive a list of statements of objects in an image.
[Task]
Your task is to rephrase each line of statement into a question following the below question template.
In specific, extract attributes of the object to fill in the attribute slot of the template to form questions. 
DO NOT RESPOND WITH ANYTHING ELSE.  DO NOT CHANGE THE QUESTION TEMPLATE.

[Template]:
Could you tell me all the objects that (ATTRIBUTE SLOT) in the image?

[Response Format]:
Could you tell me all the objects that are yellow in the image? 
Could you tell me all the objects that are parked near a marketplace in the image?
Could you tell me all the objects that are made of wood in the image?
Could you tell me all the objects that are walking down the street in the image?
Could you tell me all the objects that have tablecloths on them on the street in the image?

Now complete the following: 
[Statements]:
{statement}
[Response]:
"""


PROMPT_TEMPLATE_CHECK_EXIST_ENTITY = """
You are given a statement and a question.
[Task]
Your task is to answer the question based on the statement. The statement is about some objects. The question is to ask whether some specific object exists.
1. Your response should be limited to one of the following two choices: "Yes"/"No".
2. Note that instances of a certain category can also belong to its super-categories. For example, a baseball is a subclass of the sports ball.
3. Note that the table is equivalent to the dining table here.
4. DO NOT RESPOND WITH ANYTHING ELSE.

[Response Format]
Yes/No

Now complete the following:
[Statement]:
{statement}
[Question]:
Is there a {object} in the statement?
[Response]:
"""

PROMPT_TEMPLATE_REFINE_ENTITY = """
You are given a query, a passage and supplementary information.
[Task]
You are required to correct and output the refined passage in a fluent and natural style, following these rules:
1. Correct the sentences in the passage if they are inconsistent with the supplementary information. Remove the objects that are confirmed to not exist in the supplementary information.
2. Do not modify correct sentences and introduce additional information.
3. When giving refined passage, also pay attention to the given query. The refined passage should be a reasonable answer to the query.
4. Note the dining table is equivalent to the table.
Output only the corrected passage, without introducing extra contents.

Here are examples:
[Query]:
Is there a snowboard in the image?
[Passage]:
Yes, there is a snowboard in the image. The image shows a person skiing down a snow-covered slope.
[Supplementary Information]:
There is a snowboard. 
There is a person.
There is 
[Response]: 
Yes, there is a snowboard in the image. The image shows a person skiing down a snow-covered slope.
[Query]:
Is there a sports ball in the image?
[Passage]:
Yes, there is a sports ball in the image, and it appears to be a soccer ball.
[Supplementary Information]:
There is no sports ball in the image.
[Response]: 
No, there is no sports ball in the image.
[Query]:
Is there a ball in the image?
[Passage]:
Yes, there is a ball in this image.
[Supplementary Information]:
There is no ball in the image.
[Response]: 
No, there is no ball in the image.
[Query]:
Describe this image.
[Passage]:
The image features a brown dog, leaping into the air to catch a frisbee. The dog is in the process of jumping, with its front paws in the air and its back legs stretched out behind it. The frisbee is visible in the air, close to the dog's mouth, as it attempts to catch it.
There are several other people in the scene, with some standing closer to the dog and others further away. A car is parked in the background, and a few chairs can be seen scattered around the area. The scene appears to be a backyard setting, with the dog and its owner enjoying a fun outdoor activity.
[Supplementary Information]:
There is no chair.
There is no people.
There is a car.
There is a dog.
There is a frisbee.
[Response]:
The image captures a brown dog leaping into the air to catch a frisbee. The dog is in the process of jumping, with its front paws in the air and its back legs stretched out behind it. The frisbee is visible in the air, close to the dog's mouth, as it attempts to catch it. 
Surrounding the dog, There are no people in the scene. A car is parked in the background, and there are no chairs scattered around the area. The scene appears to be a backyard setting, with the dog and its owner enjoying a fun outdoor activity.

Now complete the following:
[Query]:
{query}
[Passage]:
{passage}
[Supplementary Information]:
{sup_info}
[Response]:
"""


SYS_PROMPT_REWRITER = "You are a language assistant that helps to rewrite a passage according to instructions."
SYS_PROMPT_EXTRACTOR = "You are a language assistant that helps to extract information from given sentences."
SYS_PROMPT_REPHRASER = "You are a language assistant that helps to rephrase sentences from given sentences."
SYS_PROMPT_CHECK = "You are a language assistant that helps to check the answers to the questions."
SYS_PROMPT_ANSWER = "You are a language assistant that helps to answer the question according to instructions."
SYS_PROMPT_REFINER = "You are a language assistant that helps to refine a passage according to instructions."


def get_response(prompt, sys_prompt, temperature=0.2, max_tokens=1024, ):
    content = prompt
    cnt = 1
    while True:
        # print("Cnt: ", cnt)
        try:
            response =  client.chat.completions.create(
            
                model='gpt-3.5-turbo-0125',
                messages=[{
                    'role': 'system',
                    'content': sys_prompt,
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

        cnt += 1
        if cnt > 3:
            print("Network Error!")

    res = response.choices[0].message.content
    return res 
    

def get_rewritten(sample):
    prompt = PROMPT_TEMPLATE_PREPROCESS.format(text=sample['input_desc'])
    rewritten_passage = get_response(prompt, sys_prompt=SYS_PROMPT_REWRITER)
    rew_split_sents = get_split_sents(rewritten_passage)
    orig_split_sents = get_split_sents(sample['input_desc'])
    sample['split_sents'] = rew_split_sents
    sample['orig_split_sents'] = orig_split_sents
    return sample
   

def get_target(sample):
    """
    extract target entities in each sentence
    """
    extracted_entities = []
    set_entities = set()                                                                    # merge the same in differen sents
    for sent in sample["split_sents"]:                                                      # describe the image: extract entities from each sent
        prompt = PROMPT_TEMPLATE_TARGET.format(sent=sent)
        entity_str = get_response(prompt, sys_prompt=SYS_PROMPT_EXTRACTOR)
        entity_str = entity_str.strip().split('.')                                          # list
        extracted_entities.append(entity_str)                                               # one sent to one list of entities
        [set_entities.add(item) for item in entity_str]

    print("Entity: ", set_entities)
    sample["split_sent_entity"] = extracted_entities
    sample["dialogues"] = [{"entity": entity} for entity in set_entities]
    
    return sample


def remove_duplicates(res):
    qs_set = set()
    output = []
    for s in res:
        qs, ent = s
        if qs in qs_set:
            continue
        else:
            output.append(s)
            qs_set.add(qs)
    return output


def remove_identity_exposion(res, entity):
    # remove questions that implicitly declare the identity of the object.
    output = []
    for s in res:
        qs, ans = s
        if " is a " in qs or " is an " in qs or " be a " in qs or " be an " in qs:
            continue
        else:
            qs = qs.replace('.', '')
            qs = qs.replace(',', '')
            qs_list = qs.split()
            if entity in qs_list or entity+"s" in qs_list:
                # mention the object in qs
                continue
            else:
                output.append(s)
    return output


def remove_irrelevant(res):
    # remove irrelevant questions
    output = []
    for s in res:
        qs, ent = s
        if "The object" not in qs and "the object" not in qs:
            continue
        else:
            output.append(s)
    return output


def get_question_general(sample):
    question_template = "Could you please describe the {entity} in the picture? Provide as much detailed information as possible, including its attributes, location, or state."
    
    for dialogue in sample['dialogues']:
        entity = dialogue["entity"]
        questions = [[question_template.format(entity=entity), entity]]
        questions = [s for s in questions if len(s)==2]
        questions = remove_duplicates(questions)
        dialogue["generated_general_questions"] = questions
    
    return sample


def get_mllm_answers(sample, question_key="generated_general_questions", answer_key="answers", additional_sample_counts=0, history=None):
    """
        Generate mllm answers to the questions.
        question_key = ["generated_general_questions", "generated_torture_questions"]
    """
    img_path = sample["img_path"]

    for dialogue in sample['dialogues']:
        gen_qs = dialogue[question_key]
        cur_answers = []
        for cur_qs in gen_qs:
            qs, entity = cur_qs
            answer = get_llava_output(img_path, qs, history=history)
            cur_answers.append(answer.strip())
            for _ in range(additional_sample_counts):                       # get sample answers
                answer = get_llava_output(img_path, qs, do_sample=True, history=history)
                cur_answers.append(answer.strip())
        dialogue[answer_key] = cur_answers

    return sample


def get_question_torture(sample):
    """
        Generate torture questions about the entities in the image
    """

    torture_list = []

    for dialogue in sample['dialogues']:
        entity = dialogue["entity"]
        ans = dialogue['answers']
        gen_qs = dialogue['generated_general_questions']

        if len(gen_qs) == 0:
            torture_list.append([])
            continue
        cur_torture = []
       
        for cur_ans in ans:
            
            prompt_torture = PROMPT_TEMPLATE_FACTOR_ATTRIBUTE.format(sent=cur_ans, entity=entity)
            torture_qs = get_response(prompt_torture, sys_prompt=SYS_PROMPT_EXTRACTOR)
            torture_qs = torture_qs.splitlines()
            torture_qs = [item.split('&') for item in torture_qs if item.lower() != 'none']                               
            torture_qs = [[item[1], item[0]] for item in torture_qs if len(item)==2]      # reverse, (question, answer)
            # torture_qs = remove_duplicates(torture_qs)
            torture_qs = remove_identity_exposion(torture_qs, entity)           # remove the claims that declare the identity of the object
            torture_qs = remove_irrelevant(torture_qs)

            cur_torture.extend(torture_qs)
            cur_torture = remove_duplicates(cur_torture)
            if len(cur_torture) > 5:                        # at least 5 questions
                # torture_qs = torture_qs[:10]
                break
            
        statement = '\n'.join([qs[0] for qs in cur_torture])
        prompt_rephrase = PROMPT_TEMPLATE_REPHRASE_TORTURE_2.format(statement=statement)
        rephrased_qs = get_response(prompt_rephrase, sys_prompt=SYS_PROMPT_REPHRASER)
        rephrased_qs = rephrased_qs.splitlines()
        # print("rephrased_qs: ", len(rephrased_qs), rephrased_qs)
        # print("cur_torture: ", len(cur_torture), cur_torture)
        cur_torture = [[rephrased_qs[idx], cur_torture[idx][1]] for idx in range(min(len(cur_torture), len(rephrased_qs)))]
        dialogue["generated_torture_questions"] = cur_torture
        
    return sample


def get_hallucination_check(sample):
    def string_match(entity, s):
        s = s.replace('.', '')
        s = s.replace(',', '')
        return entity in s.split()

    # nli_label_mapping={
    #     "Yes":1,
    #     "No": 0,
    # }
    query = sample["query"]
    for dialogue in sample["dialogues"]:
        entity = dialogue["entity"]
        general_questions = dialogue["generated_general_questions"]             # 2-d list
        answers = dialogue["answers"]
        torture_questions = dialogue["generated_torture_questions"]
        torture_answers = dialogue['answers_torture']

        predict_labels = []
        for idx, (torture_qs, torture_ans) in enumerate(zip(torture_questions, torture_answers)):
            qs, _ = torture_qs
            if string_match(entity, torture_ans):
                print(idx, "Easy Match!")
                predict_labels.append("Yes")
                continue

            # prompt = PROMPT_TEMPLATE_CHECK_EXIST_ENTITY_1.format(question=qs, golden_answer=entity, answer=torture_ans)
            prompt = PROMPT_TEMPLATE_CHECK_EXIST_ENTITY.format(statement=torture_ans, object=entity)
            response = get_response(prompt, sys_prompt=SYS_PROMPT_ANSWER, temperature=0.0)            # check needs accuracy
            response = response.strip()
            predict_labels.append(response)

        dialogue["NLI"] = predict_labels
        # nli_value = 0 
        # for label in predict_labels:
        #     nli_value += nli_label_mapping.get(label, 0)
        # if nli_value >= 1 or len(predict_labels) == 0:
        #     dialogue['entity_label'] = "Yes"
        # else:
        #     dialogue['entity_label'] = "No"
            
    return sample



def get_refinement(sample):
    nli_label_mapping={
        "Yes":1,
        "No": 0,
    }
    threshold = 0.4
    query = sample["query"]
    passage = sample["input_desc"]
    sup_info_positive = []
    sup_info_negative = []

    for dialogue in sample["dialogues"]:
        predict_labels = dialogue["NLI"]
        nli_value = 0 
        for label in predict_labels:
            nli_value += nli_label_mapping.get(label, 0)
        
        r = nli_value/len(predict_labels) if len(predict_labels) > 0 else 0
        
        if r >= threshold:
            dialogue['entity_label'] = "Yes"
        else:
            dialogue['entity_label'] = "No"

        entity = dialogue["entity"]
        if dialogue["entity_label"] == "Yes":
            sup_info_positive.append("There is a {entity}.".format(entity=entity))
        elif dialogue["entity_label"] == "No":
            sup_info_negative.append("There is no {entity}.".format(entity=entity))
        else:
            print("Check Error!")

    sup_info = "\n".join(sup_info_negative)+ "\n" +  "\n".join(sup_info_positive)
    # print(sup_info)
    prompt = PROMPT_TEMPLATE_REFINE_ENTITY.format(query=query, passage=passage, sup_info=sup_info)
    refiend_passage = get_response(prompt, sys_prompt=SYS_PROMPT_REFINER)
    sample["output"] = refiend_passage.strip()
    return sample


def get_llava_output(img_path, question, do_sample=False):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in question:
        if model.config.mm_use_im_start_end:
            question = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
        else:
            question = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
    else:
        if model.config.mm_use_im_start_end:
            question = image_token_se + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question


    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # prompt = "USER: <image>\n%s\nASSISTANT:" % question.strip()
    # print("Prompt: ", prompt)

    images = [Image.open(img_path)]
    
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample= do_sample,
            temperature=1.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


def get_split_sents(passage):
        doc = nlp(passage)
        split_sents = list(doc.sents)
        split_sents = [sent.text.strip() for sent in split_sents]
        return split_sents


def detect_non_exist(text):
    if text.find('.') != -1:
        text = text.split('.')[0]

    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        return True
    else:
        return False


def main(type):
    image_dir = "path/to/val2014"                                                 # COCO val2014 image dir
    question_file = "./dataset/pope_coco/coco_50_pope_%s.json" % (type)        # question file
    output_file = "./result/logs_llava_pope/answer_%s_logic_check.json" % (type)           # output file
    torture_log = "./result/logs_llava_pope/torture_%s_%d.json"                        # log file

    questions = [json.loads(q) for q in open(question_file, 'r')]
    answers = []

    tmp = 300
    for idx, question in enumerate(questions):
        if idx >= tmp:          
            break

        print("-"*20 + "Idx = %d"%(idx) + "-"*20)
        qid = question["question_id"]
        image_filename = question["image"]
        text = question["text"]
        label = question["label"]
        temp_image_path = os.path.join(image_dir, image_filename)

        claim = get_llava_output(temp_image_path, text)
        sample = {  
            'query': text,
            'img_path': temp_image_path,
            'input_desc': claim,
        }

        if detect_non_exist(claim):
            sample["output"] = claim
        else:
            sample = get_rewritten(sample)                                                                      # rewrite and split sents
            sample = get_target(sample)                                                                         # object extraction
            sample = get_question_general(sample)                                                               # object-to-attribute inquiring
            sample = get_mllm_answers(sample, question_key="generated_general_questions", answer_key="answers", additional_sample_counts=2)
            sample = get_question_torture(sample)                                                               # attribute-to-object inquiring
            sample = get_mllm_answers(sample, question_key="generated_torture_questions", answer_key="answers_torture", additional_sample_counts=0)
            sample = get_hallucination_check(sample)                                                            # logical closed loop checking
            sample = get_refinement(sample)                                                                     # object hallucination mitigation
        
        print(sample)
        final_answer = sample["output"]
        answer = {"quesion_id": qid, "question": text, "answer": final_answer}
        answers.append(answer)
        
        with open(torture_log % (type, idx), 'w') as f:
            json_sample = json.dumps(sample, indent=4)
            f.write(json_sample+"\n")


    with open(output_file, 'w') as f:
        for answer in answers:
            json_str = json.dumps(answer)
            f.write(json_str + "\n")


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    for type in ["adversarial"]:               # "adversarial", "popular", "random"
        print("-"*30 + type + "-"*30)
        main(type)