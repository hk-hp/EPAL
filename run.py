import importlib
import torch
import logging
import argparse
import json
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.conf.DocEE_conf import DocEEConfig

importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def get_config() -> DocEEConfig:
    config = DocEEConfig()
    config.model_save_name = 'exp_du3'
    config.node_size = 512
    config.max_epochs = 200
    config.data_loader_shuffle = True
    config.model_name = "chinese_roberta_wwm_ext"
    config.device = torch.device('cuda')
    config.max_len = 510

    config.gradient_accumulation_steps = 32
    config.temperature = 0.5
    config.data_name = 'ChFinAnn' # ChFinAnn, DuEE_Fin

    return config


def run():
    config = get_config()
    logging.info('save_name = {}'.format(config.model_save_name))
    dee_pro = DocEEProcessor(config.data_name)
    dee_pre = DocEEPreparer(config=config, processor=dee_pro)
    train_loader, dev_loader, test_loader = dee_pre.get_loader_for_flattened_fragment_before_event()
    metric = DocEEMetric(preparer=dee_pre)
    model = DocEEProxyNodeModel(config=config, preparer=dee_pre)

    # 增加词汇表中的25个大写字母
    model.language_model.resize_token_embeddings(new_num_tokens=len(dee_pre.tokenizer.vocab) + 26)
    
    model = torch.load('Result\exp_ch3\exp_ch3_58.pkl')
    model.to(config.device)

    optimizer = BasicOptimizer(config=config, model=model)
    # optimizer.load_optim('optimizer.pkl')

    # torch.save([0], 'loss.pt')

    trainer = DocEETrainer(config=config,
                        model=model,
                        optimizer=optimizer,
                        preparer=dee_pre,
                        metric=metric,
                        train_loader=train_loader,
                        dev_loader=dev_loader,
                        test_loader=test_loader,
                        )
    # trainer.train()
    score_result, _ = trainer.eval()


if __name__ == '__main__':
    run()