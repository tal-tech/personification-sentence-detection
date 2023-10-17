import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'module/text_classification'))
from module.text_classification.inference import TextClassifier


config = {
        'checkpoint_lst':[os.path.join(root, 'model/rhetoric_model/Niren_PretrainedBert_5e-06_48_None.pt')],
        'use_bert': True,
        'embd_path': os.path.join(root, 'model/word_embedding/tencent_small'),
        'model_config_lst': [{
            'is_state': False,
            'model_name': 'bert',
            'pretrained_model_path': os.path.join(root, 'model/bert_chinese_wwm_ext_pytorch')
        }],
        'max_seq_len': 115,
        'need_mask': True
}


if __name__ == "__main__":
    sent_list = [
        '风儿轻轻地拂过我的脸庞，仿佛在跟我耳语。',
    ]
    pretrained_model_path = config['model_config_lst'][0]['pretrained_model_path']
    model = TextClassifier(config['embd_path'], config['checkpoint_lst'], config['model_config_lst'],
                           pretrained_model_path)
    max_seq_len =config['max_seq_len'] if 'max_seq_len' in config else 80
    need_mask = config['need_mask'] if 'need_mask' in config else False
    pred_list, proba_list = model.predict_all_mask(sent_list, max_seq_len=max_seq_len, max_batch_size=20,
                                                   need_mask=need_mask)
    pos_sent_list = [sent_list[i] for i in range(len(pred_list)) if pred_list[i] == 1]
    # print(pred_list, proba_list)
    # print("拟人数目", len(pos_sent_list))
    # for sent in pos_sent_list:
    #     print(sent)
    if len(pos_sent_list) == 1:
        print("是拟人")
        print(pos_sent_list)
        
    else:
        print("不是拟人")
        print(sent_list)

