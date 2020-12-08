import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

from transformers import BertPreTrainedModel, BertModel
from transformers import add_start_docstrings

from code.loss_function import FocalLoss, ContrastiveLoss


def getSentenceEmbedding(hidden_state_list, batch_index_list):
        sent_embedding1 = None
        sent_embedding2 = None
        for last_hidden_state in hidden_state_list:
            embedding1 = []
            embedding2 = []
            for i in range(len(last_hidden_state)):
                last_hidden_state1 = last_hidden_state[i][1:batch_index_list[i][0]]
                last_hidden_state2 = last_hidden_state[i][batch_index_list[i][0]:batch_index_list[i][1]]

                # 词向量取mean生成句向量
                sentence_1 = last_hidden_state1.mean(0)
                sentence_2 = last_hidden_state2.mean(0)

                # 词向量取max生成句向量
                # sentence_1, _ = last_hidden_state1.max(0)
                # sentence_2, _ = last_hidden_state2.max(0)
                embedding1.append(sentence_1)
                embedding2.append(sentence_2)

            # sent_embedding1.append(torch.stack((embedding1), 0))
            if sent_embedding1 is not None:
                sent_embedding1 = sent_embedding1 + torch.stack((embedding1), 0)
                sent_embedding2 = sent_embedding2 + torch.stack((embedding2), 0)
            else:
                sent_embedding1 = torch.stack((embedding1), 0)
                sent_embedding2 = torch.stack((embedding2), 0)
        # print(sent_embedding1.size())
        sent_embedding1 = sent_embedding1 / 4
        sent_embedding2 = sent_embedding2 / 4

        return sent_embedding1, sent_embedding2



@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 30)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=0)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        # logits = self.softmax(logits)


        # print(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCELoss()
                loss = loss_fct(logits, labels.float())
                # loss = loss_fct(logits.view(-1, 30), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertForSequenceClassification_v2(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertForSequenceClassification_v2, self).__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size * 5, 30)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=0)

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        # 获得batch中每个sentence的sep_a和sep_b索引
        batch_size = len(input_ids)
        batch_index_list = []
        for whole_sentence_tensor in input_ids:  # 依次处理batch中每一个句子向量
            sep_a, sep_b = 0, 0
            for j in range(len(whole_sentence_tensor) - 1, 0, -1):  # 找到SEP的index
                if whole_sentence_tensor[j] == 102:
                    sep_a = j
                    break
            for j in range(sep_a - 1, 0, -1):
                if whole_sentence_tensor[j] == 102:
                    sep_b = j
                    break
            batch_index_list.append([sep_b, sep_a])

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sent_sentence1 = []
        sent_sentence2 = []

        last_hidden_state = outputs[0]
        for i in range(len(last_hidden_state)):
            last_hidden_state1 = last_hidden_state[i][1:batch_index_list[i][0]]
            last_hidden_state2 = last_hidden_state[i][batch_index_list[i][0]:batch_index_list[i][1]]

            # 词向量取mean生成句向量
            sentence_1 = last_hidden_state1.mean(0)
            sentence_2 = last_hidden_state2.mean(0)

            # 词向量取max生成句向量
            # sentence_1, _ = last_hidden_state1.max(0)
            # sentence_2, _ = last_hidden_state2.max(0)
            sent_sentence1.append(sentence_1)
            sent_sentence2.append(sentence_2)

        sent_embedding1 = torch.stack((sent_sentence1), 0)
        sent_embedding2 = torch.stack((sent_sentence2), 0)
        # print(sent_embedding2)



        difference = sent_embedding1 - sent_embedding2
        point_multi = sent_embedding1 * sent_embedding2

        # pooled_output = outputs[1]

        # sequence_output = outputs[0]
        # pooled_output = outputs[1]

        hidden_states = outputs[2]  # hidden_states: 12 layers tuples each is of (batch_size, sequence_length, hidden_size) + embedding``
        # print(seq[-1].shape, seq[-1][:, 0].shape)
        
        # we are taking zero because in my understanding that's the [CLS] token...
        # idea is to pool last 4 layers as well instead of just the last one, since it's too close to the output
        # layers, it might not be that efficient as it's more regulated by the o/p's..

        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h9 = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        all_h = torch.cat([h9, h10, h11, h12], 1)  # Also don't forget to add the last CLS token seq_op/pooled_op as you wish..
        mean_pool = torch.mean(all_h, 1)

        cls_embedding = torch.cat((mean_pool, sent_embedding1, sent_embedding2, difference.abs(), point_multi), -1)

        # cls_embedding = self.dropout(cls_embedding)

        # lin_output = F.relu(self.bn(self.linear(cls_embedding)))

        pooled_output = self.dropout(cls_embedding)
        # pooled_output = self.dropout(mean_pool)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        # logits = self.softmax(logits)


        # print(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCELoss()
                # loss = loss_fct(logits, labels.float())
                loss = loss_fct(logits.view(-1, 30), labels.view(-1))

                # question_weight = 0.3
                # answer_weight = 0.7

                # loss1 = loss_fct(logits[:,0:9], labels[:,0:9])
                # loss2 = loss_fct(logits[:,9:10], labels[:,9:10])
                # loss3 = loss_fct(logits[:,10:21], labels[:,10:21])
                # loss4 = loss_fct(logits[:,21:26], labels[:,21:26])
                # loss5 = loss_fct(logits[:,26:30], labels[:,26:30])
                # loss = question_weight * loss1 + answer_weight * loss2 + question_weight * loss3 + answer_weight * loss4 + question_weight * loss5

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)=


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertForSequenceClassification_sepBERT(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertForSequenceClassification_sepBERT, self).__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size * 6, 30)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=0)

        self.init_weights()


    def forward(self, q_input_ids=None, q_attention_mask=None, q_token_type_ids=None,
                a_input_ids=None, a_attention_mask=None, a_token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, q_labels=None):

        # # 获得batch中每个sentence的sep_a和sep_b索引
        # batch_size = len(input_ids)
        # batch_index_list = []
        # for whole_sentence_tensor in input_ids:  # 依次处理batch中每一个句子向量
        #     sep_a, sep_b = 0, 0
        #     for j in range(len(whole_sentence_tensor) - 1, 0, -1):  # 找到SEP的index
        #         if whole_sentence_tensor[j] == 102:
        #             sep_a = j
        #             break
        #     for j in range(sep_a - 1, 0, -1):
        #         if whole_sentence_tensor[j] == 102:
        #             sep_b = j
        #             break
        #     batch_index_list.append([sep_b, sep_a])

        q_outputs = self.bert(q_input_ids,
                            attention_mask=q_attention_mask,
                            token_type_ids=q_token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        a_outputs = self.bert(a_input_ids,
                            attention_mask=a_attention_mask,
                            token_type_ids=a_token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        # CLS 倒数四层的拼接
        q_hidden_states = q_outputs[2]

        # q_h12 = q_hidden_states[-1][:, 0].reshape((-1, 1, 768))
        # q_h11 = q_hidden_states[-2][:, 0].reshape((-1, 1, 768))
        # q_h10 = q_hidden_states[-3][:, 0].reshape((-1, 1, 768))
        # q_h9 = q_hidden_states[-4][:, 0].reshape((-1, 1, 768))

        q_h12 = q_hidden_states[-1][:, 0].reshape((-1, 1, 1024))
        q_h11 = q_hidden_states[-2][:, 0].reshape((-1, 1, 1024))
        q_h10 = q_hidden_states[-3][:, 0].reshape((-1, 1, 1024))
        q_h9 = q_hidden_states[-4][:, 0].reshape((-1, 1, 1024))

        q_all_h = torch.cat([q_h9, q_h10, q_h11, q_h12], 1)  # Also don't forget to add the last CLS token seq_op/pooled_op as you wish..
        q_mean_pool = torch.mean(q_all_h, 1)

        a_hidden_states = a_outputs[2]

        # a_h12 = a_hidden_states[-1][:, 0].reshape((-1, 1, 768))
        # a_h11 = a_hidden_states[-2][:, 0].reshape((-1, 1, 768))
        # a_h10 = a_hidden_states[-3][:, 0].reshape((-1, 1, 768))
        # a_h9 = a_hidden_states[-4][:, 0].reshape((-1, 1, 768))

        a_h12 = a_hidden_states[-1][:, 0].reshape((-1, 1, 1024))
        a_h11 = a_hidden_states[-2][:, 0].reshape((-1, 1, 1024))
        a_h10 = a_hidden_states[-3][:, 0].reshape((-1, 1, 1024))
        a_h9 = a_hidden_states[-4][:, 0].reshape((-1, 1, 1024))

        a_all_h = torch.cat([a_h9, a_h10, a_h11, a_h12], 1)  # Also don't forget to add the last CLS token seq_op/pooled_op as you wish..
        a_mean_pool = torch.mean(a_all_h, 1)


        # 句向量的拼接
        q_last_hidden_state = q_outputs[0]
        a_last_hidden_state = a_outputs[0]

        q_sent_sentence = []
        a_sent_sentence = []

        for i in range(len(q_last_hidden_state)):

            # 词向量取mean生成句向量
            sentence_1 = q_last_hidden_state[i].mean(0)
            sentence_2 = a_last_hidden_state[i].mean(0)

            # 词向量取max生成句向量
            # sentence_1, _ = last_hidden_state1.max(0)
            # sentence_2, _ = last_hidden_state2.max(0)
            q_sent_sentence.append(sentence_1)
            a_sent_sentence.append(sentence_2)

        q_sent_embedding = torch.stack((q_sent_sentence), 0)
        a_sent_embedding = torch.stack((a_sent_sentence), 0)

        difference = q_sent_embedding - a_sent_embedding
        point_multi = q_sent_embedding * a_sent_embedding

        cls_embedding = torch.cat((q_mean_pool, a_mean_pool, q_sent_embedding, a_sent_embedding, difference.abs(), point_multi), -1)


        pooled_output = self.dropout(cls_embedding)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        # logits = self.softmax(logits)


        # print(logits)

        outputs = (logits,) + q_outputs[2:]  # add hidden states and attention if they are here

        if q_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), q_labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCELoss()
                loss = loss_fct(logits, q_labels.float())
                # loss = loss_fct(logits.view(-1, 30), q_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertForSequenceClassification_v3(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertForSequenceClassification_v3, self).__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, 30)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=0)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)


        last_hidden_state = outputs[0]
        hidden_states = outputs[2] 

        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h9 = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        all_h = torch.cat([h9, h10, h11, h12], 1)  # Also don't forget to add the last CLS token seq_op/pooled_op as you wish..
        mean_pool = torch.mean(all_h, 1)

        sent_sentence1 = []
        for i in range(len(last_hidden_state)):
            # 词向量取mean生成句向量
            sentence_1 = last_hidden_state[i].mean(0)

            sent_sentence1.append(sentence_1)

        sequence_embedding = torch.stack((sent_sentence1), 0)

        cls_embedding = torch.cat((mean_pool, sequence_embedding), -1)

        pooled_output = self.dropout(cls_embedding)
        # pooled_output = self.dropout(mean_pool)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        # logits = self.softmax(logits)


        # print(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCELoss()
                loss = loss_fct(logits, labels.float())
                # loss = loss_fct(logits.view(-1, 30), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)=


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertForSequenceClassification_v4(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertForSequenceClassification_v4, self).__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 7, 30)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=0)

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        # 获得batch中每个sentence的sep_a和sep_b索引
        batch_size = len(input_ids)
        batch_index_list = []
        for whole_sentence_tensor in input_ids:  # 依次处理batch中每一个句子向量
            sep_a, sep_b = 0, 0
            for j in range(len(whole_sentence_tensor) - 1, 0, -1):  # 找到SEP的index
                if whole_sentence_tensor[j] == 102:
                    sep_a = j
                    break
            for j in range(sep_a - 1, 0, -1):
                if whole_sentence_tensor[j] == 102:
                    sep_b = j
                    break
            batch_index_list.append([sep_b, sep_a])

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sent_sentence1 = []
        sent_sentence2 = []

        last_hidden_state = outputs[0]
        for i in range(len(last_hidden_state)):
            last_hidden_state1 = last_hidden_state[i][1:batch_index_list[i][0]]
            last_hidden_state2 = last_hidden_state[i][batch_index_list[i][0]:batch_index_list[i][1]]

            # 词向量取mean生成句向量
            sentence_1 = last_hidden_state1.mean(0)
            sentence_2 = last_hidden_state2.mean(0)

            # 词向量取max生成句向量
            # sentence_1, _ = last_hidden_state1.max(0)
            # sentence_2, _ = last_hidden_state2.max(0)
            sent_sentence1.append(sentence_1)
            sent_sentence2.append(sentence_2)

        sent_embedding1 = torch.stack((sent_sentence1), 0)
        sent_embedding2 = torch.stack((sent_sentence2), 0)

        difference = sent_embedding1 - sent_embedding2
        point_multi = sent_embedding1 * sent_embedding2


        hidden_states = outputs[2]  
        hidden_sent_embedding1, hidden_sent_embedding2 = getSentenceEmbedding([hidden_states[-1],hidden_states[-2],hidden_states[-3],hidden_states[-4]], batch_index_list)


        h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        h9 = hidden_states[-4][:, 0].reshape((-1, 1, 768))

        all_h = torch.cat([h9, h10, h11, h12], 1)  # Also don't forget to add the last CLS token seq_op/pooled_op as you wish..
        mean_pool = torch.mean(all_h, 1)

        cls_embedding = torch.cat((mean_pool, sent_embedding1, sent_embedding2, difference.abs(), point_multi, hidden_sent_embedding1, hidden_sent_embedding2), -1)

        # cls_embedding = self.dropout(cls_embedding)

        # lin_output = F.relu(self.bn(self.linear(cls_embedding)))

        pooled_output = self.dropout(cls_embedding)
        # pooled_output = self.dropout(mean_pool)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        # logits = self.softmax(logits)


        # print(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # loss_fct = CrossEntropyLoss()
                loss_fct = BCELoss()
                loss = loss_fct(logits, labels.float())
                # loss = loss_fct(logits.view(-1, 30), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)=



@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertForSeqClsPlusMean(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertForSeqClsPlusMean, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 5, 30)
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        # 获得batch中每个sentence的sep_a和sep_b索引
        batch_size = len(input_ids)
        batch_index_list = []
        for whole_sentence_tensor in input_ids:  # 依次处理batch中每一个句子向量
            sep_a, sep_b = 0, 0
            for j in range(len(whole_sentence_tensor) - 1,0, -1):  # 找到SEP的index
                if whole_sentence_tensor[j] == 102:
                    sep_a = j
                    break
            for j in range(sep_a - 1, 0, -1):
                if whole_sentence_tensor[j] == 102:
                    sep_b = j
                    break
            batch_index_list.append([sep_b, sep_a])

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sent_sentence1 = []
        sent_sentence2 = []

        last_hidden_state = outputs[0]
        for i in range(len(last_hidden_state)):
            last_hidden_state1 = last_hidden_state[i][1:batch_index_list[i][0]]
            last_hidden_state2 = last_hidden_state[i][batch_index_list[i][0]:batch_index_list[i][1]]

            # 词向量取mean生成句向量
            sentence_1 = last_hidden_state1.mean(0)
            sentence_2 = last_hidden_state2.mean(0)

            # 词向量取max生成句向量
            # sentence_1, _ = last_hidden_state1.max(0)
            # sentence_2, _ = last_hidden_state2.max(0)
            sent_sentence1.append(sentence_1)
            sent_sentence2.append(sentence_2)
        
        sent_embedding1 = torch.stack((sent_sentence1), 0)
        sent_embedding2 = torch.stack((sent_sentence2), 0)
        # print(sent_embedding2)

        difference = sent_embedding1 - sent_embedding2
        point_multi = sent_embedding1 * sent_embedding2

        pooled_output = outputs[1]

        cls_embedding = torch.cat((pooled_output, sent_embedding1, sent_embedding2, difference.abs(), point_multi), -1)
        # print(cls_embedding.size())

        logits = self.classifier(self.dropout(cls_embedding))
        logits = self.sigmoid(logits)

        # logits = self.softmax(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = BCELoss()
                loss = loss_fct(logits, labels.float())
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # loss_fct = FocalLoss(class_num=self.num_labels, alpha=0.5)
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """, )
class BertSiamese(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config):
        super(BertSiamese, self).__init__(config)
        self.num_labels = config.num_labels
        self.embed_dim = 50

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 4, self.config.num_labels)

        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)

        self.loss_weight = 0.1
        self.init_weights()

    def select_last_tensor(self, a, b):
        # a: 3维 tensor
        # b: 1维 tensor
        _list = []
        for i, item in enumerate(a):
            _list.append(item[b[i]].unsqueeze(0))
        return torch.cat(_list, 0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        batch_size = labels.size()[-1]
        half_batch_size = batch_size // 2
        outputs1 = self.bert(input_ids[::2],
                             attention_mask=attention_mask[::2],
                             token_type_ids=token_type_ids[::2],
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds)

        outputs2 = self.bert(input_ids[1::2],
                             attention_mask=attention_mask[1::2],
                             token_type_ids=token_type_ids[1::2],
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds)

        # sent_embedding1= outputs1[1]
        # sent_embedding2= outputs2[1]

        last_hidden_state1 = outputs1[0]
        last_hidden_state2 = outputs2[0]

        #################################################################
        # # 平均，作为句子 embedding
        sent_embedding1 = last_hidden_state1.mean(1)
        sent_embedding2 = last_hidden_state2.mean(1)

        # # lstm
        # output1, _ =  self.rnn(last_hidden_state1, None)
        # output2, _ =  self.rnn(last_hidden_state2, None)

        # sent_embedding1= output1[:,-1,:]
        # sent_embedding2= output2[:,-1,:]

        # dynamic lstm
        # sent_lengths1 = attention_mask[::2].sum(-1)
        # sent_lengths2 = attention_mask[1::2].sum(-1)
        # last_hidden_state1 = nn.utils.rnn.pack_padded_sequence(last_hidden_state1,
        #                                                        sent_lengths1,
        #                                                        batch_first=True,
        #                                                        enforce_sorted=False)
        # last_hidden_state2 = nn.utils.rnn.pack_padded_sequence(last_hidden_state2,
        #                                                        sent_lengths2,
        #                                                        batch_first=True,
        #                                                        enforce_sorted=False)

        # output1, _ =  self.rnn(last_hidden_state1, None)
        # output2, _ =  self.rnn(last_hidden_state2, None)

        # output1, _ = nn.utils.rnn.pad_packed_sequence(output1, batch_first=True)
        # output2, _ = nn.utils.rnn.pad_packed_sequence(output2, batch_first=True)

        # sent_embedding1= self.select_last_tensor(output1,sent_lengths1-1)
        # sent_embedding2= self.select_last_tensor(output2,sent_lengths2-1)

        ###########################################################################
        difference = sent_embedding1 - sent_embedding2
        point_multi = sent_embedding1 * sent_embedding2

        # cls_embedding = torch.cat((sent_embedding1, sent_embedding2), -1)
        cls_embedding = torch.cat((sent_embedding1, sent_embedding2, difference.abs(), point_multi), -1)

        logits = self.classifier(self.dropout(cls_embedding))

        labels = labels[::2]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = ContrastiveLoss(margin=1)
                contrast_loss, distances = loss_fct(sent_embedding1, sent_embedding2, labels)

                loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                print(contrast_loss, cls_loss)
                loss = contrast_loss + self.loss_weight * cls_loss
                loss = cls_loss
            outputs = (loss,) + (logits,)

        return outputs  # (loss), logits, (hidden_states), (attentions)