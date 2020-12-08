import json
import os
import pandas as pd
import re
import csv


def read_prob_file(input_dir, file_nums):
    total_items = []
    for i in range(file_nums):
        input_file = input_dir + str(i+1) + ".jsonl"
        with open(input_file, encoding='utf-8') as infile:
            items = []
            for row in infile:
                row = json.loads(row)
                item = {}
                int_list = []
                item['id'] = row['id']
                prob_list = re.findall(r"\d+\.?\d*",row['labels'])
                for prob in prob_list:
                    int_list.append(float(prob))
                item['prob'] = int_list
                # print(len(int_list))
                items.append(item)
        total_items.append(items)
    return total_items


def ensemble(input_dir, file_nums, threshold, output_file):
    total_items = read_prob_file(input_dir, file_nums)
    label_list = [0, 1]
    ids = []
    labels = []
    for i in range(len(total_items[0])):
        probs = []
        for j in range(file_nums):
            probs.append(total_items[j][i]['prob'])
        positive_count = 0
        negative_count = 0
        for prob in probs:
            if prob > threshold:
                negative_count += 1
            else:
                positive_count += 1
        if positive_count > 4:
            labels.append(label_list[1])
        else:
            labels.append(label_list[0])
        ids.append(i)
    df = pd.DataFrame()
    df['id'] = ids
    df['label'] = labels
    df.to_csv(output_file, index=False, encoding='utf-8', sep="\t")


def ensemble_test(input_dir, output_file):
    samples = pd.read_csv("/home/mhxia/BD/QA_Labeling/data/test.csv")
    ids = samples['qa_id']
    with open(input_dir, encoding='utf-8') as infile:
        items = []
        row_count = 0
        for row in infile:
            row = json.loads(row)
            item = {}
            int_list = []
            item['id'] = ids[row_count]
            row_count += 1
            prob_list = re.findall(r"\d+\.?\d*",row['labels'])
            for prob in prob_list:
                int_list.append(str(prob))
            item['label'] = int_list
            # print(len(int_list))
            items.append(item)
    with open(output_file, 'w+', encoding='utf-8') as f:
        f.write("qa_id,question_asker_intent_understanding,question_body_critical,question_conversational,question_expect_short_answer,question_fact_seeking,question_has_commonly_accepted_answer,question_interestingness_others,question_interestingness_self,question_multi_intent,question_not_really_a_question,question_opinion_seeking,question_type_choice,question_type_compare,question_type_consequence,question_type_definition,question_type_entity,question_type_instructions,question_type_procedure,question_type_reason_explanation,question_type_spelling,question_well_written,answer_helpful,answer_level_of_information,answer_plausible,answer_relevance,answer_satisfaction,answer_type_instructions,answer_type_procedure,answer_type_reason_explanation,answer_well_written\n")
        for item in items:
            f.write(str(item['id']) + ',' + ",".join(item['label']))
            f.write("\n")




def func():
    input_dir = '/home/mhxia/BD/QA_Labeling/results/baseline/pred_results_1.jsonl'
    output_file = '/home/mhxia/BD/QA_Labeling/results/baseline/result.csv'

    # input_file  = '../data/processed_data/test_some.jsonl'
    # output_file = '../data/results/result_some.csv'

    # ensemble(input_dir, file_nums=1, threshold=0.5, output_file=output_file)
    ensemble_test(input_dir, output_file)
    pass


if __name__ == '__main__':
    func()